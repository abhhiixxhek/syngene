import argparse
import json
import os
from src.ingestion.indexer import ReferenceIndexer
from src.verification.verifier import SOPVerifier

def main():
    parser = argparse.ArgumentParser(description="SOP Compliance Verification Engine")
    subparsers = parser.add_subparsers(dest="command", help="Mode of operation")
    
    # Mode 1: Ingest References
    ingest_parser = subparsers.add_parser("ingest", help="Ingest reference documents into S3")
    ingest_parser.add_argument("--files", nargs="+", required=True, help="List of PDF/DOCX files to ingest")
    ingest_parser.add_argument("--bucket", required=True, help="S3 Bucket Name")
    ingest_parser.add_argument("--region", default="us-east-1", help="AWS Region")
    
    # Mode 2: Verify SOP
    verify_parser = subparsers.add_parser("verify", help="Verify an SOP against the index")
    verify_parser.add_argument("--sop", required=True, help="Path to SOP file")
    verify_parser.add_argument("--bucket", required=True, help="S3 Bucket Name")
    verify_parser.add_argument("--region", default="us-east-1", help="AWS Region")
    verify_parser.add_argument("--output", default="gap_report.json", help="Output file for the report")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        indexer = ReferenceIndexer(bucket_name=args.bucket, region_name=args.region)
        indexer.process_and_index(args.files)
        
    elif args.command == "verify":
        verifier = SOPVerifier(bucket_name=args.bucket, region_name=args.region)
        
        # 1. Load Index
        try:
            verifier.load_reference_index()
        except Exception as e:
            print(f"Failed to load index. Ensure you have run 'ingest' first. Error: {e}")
            return

        # 2. Ingest SOP
        verifier.ingest_sop(args.sop)
        
        # 3. Verify
        print("Starting verification (this may take a minute)...")
        gaps = verifier.verify_documents()
        
        # 4. Report
        print(f"Verification Complete. Found {len(gaps)} potential gaps.")
        
        # Save JSON
        with open(args.output, 'w') as f:
            json.dump(gaps, f, indent=2)
            
        print(f"Report saved to {args.output}")
        
        # Print summary to console
        if gaps:
            print("\n--- GAP SUMMARY ---")
            for gap in gaps[:5]:
                print(f"[{gap['status']}] {gap['full_reference_text'][:100]}... (Ref: {gap['reference_source']})")
            if len(gaps) > 5:
                print(f"... and {len(gaps)-5} more.")
                
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
