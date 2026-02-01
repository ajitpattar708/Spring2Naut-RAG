import argparse
import sys
import os

# Ensure the src directory is in the path
sys.path.append(os.path.join(os.getcwd(), "src"))

from agent.orchestrator import MigrationOrchestrator

def main():
    """
    Main entry point for the Spring to Micronaut Migration Agent.
    Provides a command-line interface for the migration tool.
    """
    parser = argparse.ArgumentParser(description="Spring Boot to Micronaut Migration Agent")
    parser.add_argument("input", help="Path to the source Spring Boot project directory")
    parser.add_argument("output", help="Path to the target Micronaut project directory")
    parser.add_argument("--spring-version", default="3.4.5", help="Source Spring Boot version")
    parser.add_argument("--micronaut-version", default="4.10.8", help="Target Micronaut version")
    
    args = parser.parse_args()
    
    print("-" * 50)
    print("Agentic Migration Initialized")
    print(f"Targeting: Spring {args.spring_version} -> Micronaut {args.micronaut_version}")
    print("-" * 50)
    
    try:
        orchestrator = MigrationOrchestrator(
            spring_version=args.spring_version,
            micronaut_version=args.micronaut_version
        )
        
        report = orchestrator.migrate_project(args.input, args.output)
        
        # Output summary report
        print("\n" + "=" * 50)
        print("MIGRATION SUMMARY")
        print("=" * 50)
        print(f"Total Files Processed: {report.total_files}")
        print(f"Successfully Migrated: {report.migrated_files}")
        print(f"Failed Files: {len(report.failed_files)}")
        if report.failed_files:
            for f in report.failed_files:
                print(f"  - {f}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
