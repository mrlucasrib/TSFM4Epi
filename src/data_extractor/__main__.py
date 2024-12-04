import argparse
import logging

from data_extractor.sinan_extractor import SinanDataExtractor

def main():
    parser = argparse.ArgumentParser(description="SINAN Data Extractor")
    parser.add_argument('--path', type=str, required=False, help='Path to save the data')
    parser.add_argument('--disease-codes', type=str, nargs='+', required=False, help='List of disease codes to extract')
    parser.add_argument('--frequency', type=str, default='MS', help='Frequency of data extraction (default: MS)')
    parser.add_argument('--geographic-level', type=str, default='municipality', help='Geographic level of data extraction (default: region)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
        logging.info("Verbose logging enabled")

    extractor = SinanDataExtractor(
        path=args.path, 
        disease_codes=args.disease_codes, 
        frequency=args.frequency, 
        geographic_level=args.geographic_level
    )
    extractor.extract()

if __name__ == "__main__":
    main()