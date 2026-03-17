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
        self.last_output = ""

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
            self.last_output = result.stdout + "\n" + result.stderr
            errors = self._parse_maven_errors(self.last_output) if not success else []
            
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
            self.last_output = result.stdout + "\n" + result.stderr
            errors = self._parse_gradle_errors(self.last_output) if not success else []
            
            return success, errors
        except Exception as e:
            return False, [f"Gradle execution failed: {str(e)}"]

    def _parse_maven_errors(self, stdout: str) -> List[str]:
        """
        Extracts specific compilation error messages from Maven output.
        Fails gracefully if no specific Java errors found but build failed.
        """
        errors = []
        # Pattern to find [ERROR] lines with file info
        lines = stdout.split('\n')
        for line in lines:
            # Catch standard Java compilation errors
            if "[ERROR]" in line and (".java:" in line or "error:" in line.lower()):
                errors.append(line.strip())
            # Catch dependency resolution/missing parent POM issues
            elif "[ERROR]" in line and ("Could not find artifact" in line or "Non-resolvable parent POM" in line):
                errors.append(line.strip())
            # Catch Micronaut-specific annotation processing errors
            elif "error: Failed to extract" in line or "error: Cannot build" in line:
                errors.append(line.strip())
        
        # If no specific errors found but build failed (this logic is called after checking returncode)
        if not errors:
            # Check for generic fatal failures
            for line in lines:
                if "FATAL" in line or "BUILD FAILURE" in line:
                    errors.append(line.strip())
                    
        # Limit to avoid context bloat
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
