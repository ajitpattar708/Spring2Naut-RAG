import subprocess
import os
import re
from typing import List, Dict, Optional, Tuple

class ValidationAgent:
    """
    ValidationAgent is responsible for verifying the integrity of the migrated project.
    It attempts to build the project and extracts actionable error messages for the 
    self-refinement loop.
    """

    def __init__(self, build_tool: str = "maven"):
        self.build_tool = build_tool.lower()

    def validate(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Runs the project build and returns success status along with error logs.
        """
        if self.build_tool == "maven":
            return self._run_maven_build(project_path)
        elif self.build_tool == "gradle":
            return self._run_gradle_build(project_path)
        return False, ["Unsupported build tool"]

    def _run_maven_build(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Executes 'mvn clean compile' and captures output.
        """
        try:
            # Running with -DskipTests to focus on compilation errors first
            command = ["mvn", "clean", "compile", "-B"]
            result = subprocess.run(
                command, 
                cwd=project_path, 
                capture_output=True, 
                text=True, 
                shell=True
            )
            
            success = result.returncode == 0
            errors = self._parse_maven_errors(result.stdout) if not success else []
            
            return success, errors
        except Exception as e:
            return False, [f"Maven execution failed: {str(e)}"]

    def _run_gradle_build(self, project_path: str) -> Tuple[bool, List[str]]:
        """
        Executes './gradlew classes' or 'gradle classes' and captures output.
        """
        try:
            command = ["gradle", "classes"]
            if os.name != 'nt': # Unix-like
                if os.path.exists(os.path.join(project_path, "gradlew")):
                    command = ["./gradlew", "classes"]

            result = subprocess.run(
                command, 
                cwd=project_path, 
                capture_output=True, 
                text=True, 
                shell=True
            )
            
            success = result.returncode == 0
            errors = self._parse_gradle_errors(result.stdout) if not success else []
            
            return success, errors
        except Exception as e:
            return False, [f"Gradle execution failed: {str(e)}"]

    def _parse_maven_errors(self, stdout: str) -> List[str]:
        """
        Extracts specific compilation error messages from Maven output.
        Focuses on file paths and error descriptions.
        """
        errors = []
        # Pattern to find [ERROR] lines with file info
        lines = stdout.split('\n')
        for line in lines:
            if "[ERROR]" in line and (".java:" in line or "error:" in line.lower()):
                errors.append(line.strip())
        
        # Limit to first 10 errors to avoid overloading the LLM context
        return errors[:10]

    def _parse_gradle_errors(self, stdout: str) -> List[str]:
        """
        Extracts specific compilation error messages from Gradle output.
        """
        errors = []
        # Simple extraction of error blocks
        lines = stdout.split('\n')
        for line in lines:
            if "error:" in line.lower() or "FAILED" in line:
                errors.append(line.strip())
        return errors[:10]
