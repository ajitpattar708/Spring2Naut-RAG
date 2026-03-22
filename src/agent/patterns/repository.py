import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CorpusSection:
    name: str
    description: str
    subdirectories: List[str]


DEFAULT_SECTIONS = [
    CorpusSection(
        name="official_docs",
        description="Primary-source documents and normalized patterns derived from Micronaut/Spring official documentation.",
        subdirectories=["raw", "normalized"],
    ),
    CorpusSection(
        name="github_candidates",
        description="Curated GitHub-derived candidate patterns pending stronger validation.",
        subdirectories=["raw", "normalized"],
    ),
    CorpusSection(
        name="staged_patterns",
        description="Patterns promoted from candidate state and waiting for compile/regression validation.",
        subdirectories=["candidates", "review_notes"],
    ),
    CorpusSection(
        name="validated_patterns",
        description="Release-ready validated patterns that can feed runtime datasets.",
        subdirectories=["release", "archives"],
    ),
]


def default_manifest(section: CorpusSection) -> Dict[str, object]:
    return {
        "section": section.name,
        "description": section.description,
        "schema_version": 1,
        "status": "active",
        "subdirectories": section.subdirectories,
    }


class PatternCorpusRepository:
    def __init__(self, root: str = "corpus"):
        self.root = Path(root)

    def initialize_layout(self) -> Dict[str, object]:
        self.root.mkdir(parents=True, exist_ok=True)
        summary = {"root": str(self.root), "sections": []}

        for section in DEFAULT_SECTIONS:
            section_root = self.root / section.name
            section_root.mkdir(parents=True, exist_ok=True)

            manifest_path = section_root / "manifest.json"
            if not manifest_path.exists():
                manifest_path.write_text(json.dumps(default_manifest(section), indent=2), encoding="utf-8")

            created_subdirs = []
            for directory in section.subdirectories:
                target = section_root / directory
                target.mkdir(parents=True, exist_ok=True)
                keep_file = target / ".gitkeep"
                if not keep_file.exists():
                    keep_file.write_text("", encoding="utf-8")
                created_subdirs.append(str(target))

            summary["sections"].append(
                {
                    "name": section.name,
                    "manifest": str(manifest_path),
                    "subdirectories": created_subdirs,
                }
            )

        return summary

    def list_pattern_files(self, section_name: str) -> List[str]:
        section_root = self.root / section_name
        if not section_root.exists():
            return []

        pattern_files = []
        for path in sorted(section_root.rglob("*.json")):
            if path.name == "manifest.json":
                continue
            pattern_files.append(str(path))
        return pattern_files

    def validate_layout(self) -> List[str]:
        issues: List[str] = []

        for section in DEFAULT_SECTIONS:
            section_root = self.root / section.name
            if not section_root.exists():
                issues.append(f"Missing section directory: {section_root}")
                continue

            manifest_path = section_root / "manifest.json"
            if not manifest_path.exists():
                issues.append(f"Missing manifest: {manifest_path}")

            for directory in section.subdirectories:
                if not (section_root / directory).exists():
                    issues.append(f"Missing subdirectory: {section_root / directory}")

        return issues


def main():
    parser = argparse.ArgumentParser(description="Initialize or inspect the pattern corpus repository layout")
    parser.add_argument("--root", default="corpus", help="Corpus root directory")
    parser.add_argument("--init", action="store_true", help="Initialize the corpus directory layout")
    parser.add_argument("--validate", action="store_true", help="Validate the corpus directory layout")
    args = parser.parse_args()

    repository = PatternCorpusRepository(root=args.root)

    if args.init:
        print(json.dumps(repository.initialize_layout(), indent=2, sort_keys=True))
        return

    if args.validate:
        issues = repository.validate_layout()
        print(json.dumps({"ok": not issues, "issues": issues}, indent=2, sort_keys=True))
        return

    print(
        json.dumps(
            {
                "root": str(repository.root),
                "sections": [asdict(section) for section in DEFAULT_SECTIONS],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
