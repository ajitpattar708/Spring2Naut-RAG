import os
import shutil
import subprocess
from typing import Dict, List, Tuple


class ValidationAgent:
    """
    ValidationAgent is responsible for verifying the integrity of the migrated project.
    It attempts to build the project and extracts actionable error messages for the
    self-refinement loop.
    """

    def __init__(self, build_tool: str = "maven"):
        self.build_tool = build_tool.lower()
        self.last_output = ""
        self.last_failure_kind = ""

    def validate(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Runs the project build and returns success status along with error logs.
        """
        if self.build_tool == "maven":
            return self._run_maven_build(project_path)
        if self.build_tool == "gradle":
            return self._run_gradle_build(project_path)
        return False, ["Unsupported build tool"]

    def _run_maven_build(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Executes compile validation first, then a runtime test pass so the migration
        report does not claim success when generated tests still fail at runtime.
        """
        try:
            mvn_cmd = shutil.which("mvn")
            compile_command = [mvn_cmd, "clean", "test-compile", "-B", "-DskipTests"] if mvn_cmd else ["mvn", "clean", "test-compile", "-B", "-DskipTests"]
            test_command = [mvn_cmd, "test", "-B", "-DfailIfNoTests=false"] if mvn_cmd else ["mvn", "test", "-B", "-DfailIfNoTests=false"]
            if os.path.exists(os.path.join(project_path, "mvnw")):
                compile_command = ["./mvnw", "clean", "test-compile", "-B", "-DskipTests"]
                test_command = ["./mvnw", "test", "-B", "-DfailIfNoTests=false"]
            elif mvn_cmd is None:
                return False, ["Maven execution failed: mvn not found in PATH"]

            self.last_failure_kind = ""
            compile_result = subprocess.run(
                compile_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                shell=False,
                env=self._build_subprocess_env(),
            )

            compile_success = compile_result.returncode == 0
            self.last_output = self._sanitize_build_output(compile_result.stdout + "\n" + compile_result.stderr)
            if not compile_success:
                self.last_failure_kind = "compile"
                return False, self._parse_maven_errors(self.last_output)

            test_result = subprocess.run(
                test_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                shell=False,
                env=self._build_subprocess_env(),
            )

            test_success = test_result.returncode == 0
            self.last_output = self._sanitize_build_output(test_result.stdout + "\n" + test_result.stderr)
            if not test_success:
                self.last_failure_kind = (
                    "environment" if self._is_environment_runtime_block(self.last_output) else "test"
                )
                return False, []

            return True, []
        except Exception as exc:
            return False, [f"Maven execution failed: {str(exc)}"]

    def _run_gradle_build(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Executes compile validation first, then a runtime test pass so the migration
        report does not claim success when generated tests still fail at runtime.
        """
        try:
            gradle_cmd = shutil.which("gradle")
            compile_command = [gradle_cmd, "testClasses"] if gradle_cmd else ["gradle", "testClasses"]
            test_command = [gradle_cmd, "test"] if gradle_cmd else ["gradle", "test"]
            if os.name != "nt":
                if os.path.exists(os.path.join(project_path, "gradlew")):
                    compile_command = ["./gradlew", "testClasses"]
                    test_command = ["./gradlew", "test"]
                elif gradle_cmd is None:
                    return False, ["Gradle execution failed: gradle not found in PATH and no wrapper present"]
            elif gradle_cmd is None:
                return False, ["Gradle execution failed: gradle not found in PATH and no wrapper present"]

            self.last_failure_kind = ""
            compile_result = subprocess.run(
                compile_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                shell=False,
                env=self._build_subprocess_env(),
            )

            compile_success = compile_result.returncode == 0
            self.last_output = self._sanitize_build_output(compile_result.stdout + "\n" + compile_result.stderr)
            if not compile_success:
                self.last_failure_kind = "compile"
                return False, self._parse_gradle_errors(self.last_output)

            test_result = subprocess.run(
                test_command,
                cwd=project_path,
                capture_output=True,
                text=True,
                shell=False,
                env=self._build_subprocess_env(),
            )

            test_success = test_result.returncode == 0
            self.last_output = self._sanitize_build_output(test_result.stdout + "\n" + test_result.stderr)
            if not test_success:
                self.last_failure_kind = (
                    "environment" if self._is_environment_runtime_block(self.last_output) else "test"
                )
                return False, []

            return True, []
        except Exception as exc:
            return False, [f"Gradle execution failed: {str(exc)}"]

    def _parse_maven_errors(self, stdout: str) -> List[str]:
        """
        Extracts specific compilation or dependency-resolution failures from Maven output.
        Generic BUILD FAILURE lines are intentionally ignored so environment issues do not
        trigger the self-fix loop.
        """
        errors: List[str] = []
        lines = stdout.split("\n")
        for line in lines:
            lowered = line.lower()
            if "[ERROR]" in line and (".java:" in line or "error:" in lowered):
                errors.append(line.strip())
                continue
            if "[ERROR]" in line and (
                "cannot find symbol" in lowered
                or "symbol:" in lowered
                or "location:" in lowered
                or "package " in lowered and " does not exist" in lowered
                or "incompatible types" in lowered
            ):
                errors.append(line.strip())
                continue
            if "[ERROR]" in line and ("could not find artifact" in lowered or "non-resolvable parent pom" in lowered):
                errors.append(line.strip())
                continue
            if "[ERROR]" in line and ("fatal error compiling" in lowered or "illegalargumentexception" in lowered):
                errors.append(line.strip())
                continue
            if "error: failed to extract" in lowered or "error: cannot build" in lowered:
                errors.append(line.strip())

        return errors[:10]

    def _parse_gradle_errors(self, stdout: str) -> List[str]:
        """
        Extracts specific compilation or dependency-resolution failures from Gradle output.
        Generic BUILD FAILED environment/toolchain failures are intentionally ignored so
        self-fix is only attempted for actionable source/build-file issues.
        """
        errors: List[str] = []
        lines = stdout.split("\n")
        for line in lines:
            lowered = line.lower()
            if ".java:" in line and ("error:" in lowered or "warning:" in lowered):
                errors.append(line.strip())
                continue
            if "execution failed for task" in lowered and ("compile" in lowered or "kapt" in lowered or "test" in lowered):
                errors.append(line.strip())
                continue
            if "cannot find symbol" in lowered or "symbol:" in lowered or "location:" in lowered:
                errors.append(line.strip())
                continue
            if "could not resolve all files for configuration" in lowered:
                errors.append(line.strip())
                continue
            if "dependency resolution is looking for a library compatible with jvm runtime version" in lowered:
                errors.append(line.strip())
                continue
            if "is only compatible with jvm runtime version" in lowered:
                errors.append(line.strip())
                continue
            if "could not find " in lowered and "sought in the following locations" not in lowered:
                errors.append(line.strip())
                continue
            if "could not set unknown property" in lowered:
                errors.append(line.strip())
                continue
            if "compilation failed" in lowered:
                errors.append(line.strip())

        return errors[:10]

    def _build_subprocess_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        return env

    def _is_environment_runtime_block(self, output: str) -> bool:
        lowered = str(output or "").lower()
        markers = (
            "java.net.socketexception: operation not permitted",
            "permission denied",
            "address already in use",
            "failed to bind",
            "unable to start micronaut server on *:-1",
        )
        return any(marker in lowered for marker in markers)

    def _sanitize_build_output(self, output: str) -> str:
        sanitized_lines: List[str] = []
        skip_notice = False

        for line in str(output or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("huggingface/tokenizers:"):
                skip_notice = True
                continue
            if skip_notice and (
                stripped.startswith("To disable this warning")
                or line.startswith("\t- ")
                or line.startswith("    - ")
                or not stripped
            ):
                continue

            skip_notice = False
            sanitized_lines.append(line)

        return "\n".join(sanitized_lines)
