# utils/icd10_processor.py
"""
ICD-10-CM Code Processor and Indexer
Parses and indexes ICD-10-CM codes for use by the Diagnostic Agent
"""

import re
import json
import os
from datetime import datetime
from pathlib import Path


class ICD10Processor:
    """
    Processes and indexes ICD-10-CM codes from various file formats.
    Supports parsing from HTML-formatted text files and creating searchable indices.
    """

    def __init__(self, index_dir="data/icd10_index"):
        """
        Initialize the ICD10Processor.

        Args:
            index_dir (str): Directory to store the ICD-10 index files
        """
        self.index_dir = index_dir
        self.codes_index = {}
        self.categories_index = {}
        self.metadata = {}
        os.makedirs(index_dir, exist_ok=True)

    def parse_and_load(self, file_path):
        """
        Parse ICD-10-CM data from file and load into memory index.

        Args:
            file_path (str): Path to the ICD-10-CM data file

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            if not os.path.exists(file_path):
                return (
                    False,
                    f"File not found: {file_path}",
                )

            # Detect file type and parse accordingly
            if file_path.endswith(".txt"):
                success, message = self._parse_html_text_file(file_path)
            elif file_path.endswith(".xlsx"):
                success, message = self._parse_excel_file(file_path)
            else:
                return (False, f"Unsupported file format: {file_path}")

            if success:
                # Save index to disk
                self._save_index()
                total_codes = len(self.codes_index)
                total_categories = len(self.categories_index)
                return (
                    True,
                    f"Successfully loaded {total_codes} ICD-10 codes in {total_categories} categories.",
                )
            else:
                return (False, message)

        except Exception as e:
            return (False, f"Error processing ICD-10 data: {str(e)}")

    def _parse_html_text_file(self, file_path):
        """
        Parse ICD-10-CM data from HTML-formatted text file.

        Args:
            file_path (str): Path to the HTML text file

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Extract all <p> tag contents
            pattern = r"<p>\s*(.*?)\s*</p>"
            matches = re.findall(pattern, content, re.DOTALL)

            if not matches:
                return (False, "No valid data found in the file.")

            current_category = None
            icd_code_pattern = r"^([A-Z]\d{2}[\.\d]*)\s+(.+)$"

            for match in matches:
                match = match.strip()
                if not match:
                    continue

                # Check if this is an ICD-10 code entry
                code_match = re.match(icd_code_pattern, match)
                if code_match:
                    code = code_match.group(1)
                    description = code_match.group(2).strip()

                    # Store in codes index
                    self.codes_index[code] = {
                        "code": code,
                        "description": description,
                        "category": current_category,
                        "added_date": datetime.now().isoformat(),
                    }

                    # Update category index
                    if current_category:
                        if current_category not in self.categories_index:
                            self.categories_index[current_category] = []
                        self.categories_index[current_category].append(code)
                else:
                    # This is likely a category header
                    if match and not match.startswith("DRG"):
                        current_category = match

            # Update metadata
            self.metadata = {
                "total_codes": len(self.codes_index),
                "total_categories": len(self.categories_index),
                "last_updated": datetime.now().isoformat(),
                "source_file": file_path,
            }

            return (True, f"Parsed {len(self.codes_index)} codes successfully.")

        except Exception as e:
            return (False, f"Error parsing HTML text file: {str(e)}")

    def _parse_excel_file(self, file_path):
        """
        Parse ICD-10-CM data from Excel file.

        Args:
            file_path (str): Path to the Excel file

        Returns:
            tuple: (success: bool, message: str)
        """
        try:
            import pandas as pd

            df = pd.read_excel(file_path)

            # Expected columns: 'Code', 'Description', 'Category' or similar
            code_col = None
            desc_col = None
            cat_col = None

            # Auto-detect column names
            for col in df.columns:
                col_lower = col.lower()
                if "code" in col_lower:
                    code_col = col
                elif "description" in col_lower or "desc" in col_lower:
                    desc_col = col
                elif "category" in col_lower or "cat" in col_lower:
                    cat_col = col

            if not code_col or not desc_col:
                return (False, "Could not find required columns (Code, Description) in Excel file.")

            for idx, row in df.iterrows():
                code = str(row[code_col]).strip()
                description = str(row[desc_col]).strip()
                category = str(row[cat_col]).strip() if cat_col else "Uncategorized"

                if code and code != "nan":
                    self.codes_index[code] = {
                        "code": code,
                        "description": description,
                        "category": category,
                        "added_date": datetime.now().isoformat(),
                    }

                    if category not in self.categories_index:
                        self.categories_index[category] = []
                    self.categories_index[category].append(code)

            self.metadata = {
                "total_codes": len(self.codes_index),
                "total_categories": len(self.categories_index),
                "last_updated": datetime.now().isoformat(),
                "source_file": file_path,
            }

            return (True, f"Parsed {len(self.codes_index)} codes from Excel successfully.")

        except ImportError:
            return (False, "pandas library not found. Install it with: pip install pandas openpyxl")
        except Exception as e:
            return (False, f"Error parsing Excel file: {str(e)}")

    def _save_index(self):
        """Save the ICD-10 index to disk as JSON files."""
        try:
            codes_file = os.path.join(self.index_dir, "icd10_codes.json")
            categories_file = os.path.join(self.index_dir, "icd10_categories.json")
            metadata_file = os.path.join(self.index_dir, "metadata.json")

            with open(codes_file, "w", encoding="utf-8") as f:
                json.dump(self.codes_index, f, indent=2, ensure_ascii=False)

            with open(categories_file, "w", encoding="utf-8") as f:
                json.dump(self.categories_index, f, indent=2, ensure_ascii=False)

            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Warning: Could not save index to disk: {str(e)}")

    def load_index(self):
        """Load the ICD-10 index from disk."""
        try:
            codes_file = os.path.join(self.index_dir, "icd10_codes.json")
            categories_file = os.path.join(self.index_dir, "icd10_categories.json")
            metadata_file = os.path.join(self.index_dir, "metadata.json")

            if os.path.exists(codes_file):
                with open(codes_file, "r", encoding="utf-8") as f:
                    self.codes_index = json.load(f)

            if os.path.exists(categories_file):
                with open(categories_file, "r", encoding="utf-8") as f:
                    self.categories_index = json.load(f)

            if os.path.exists(metadata_file):
                with open(metadata_file, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

            return (True, "Index loaded successfully.")
        except Exception as e:
            return (False, f"Error loading index: {str(e)}")

    def search_code(self, code):
        """
        Search for a specific ICD-10 code.

        Args:
            code (str): The ICD-10 code to search for

        Returns:
            dict: Code information or None if not found
        """
        return self.codes_index.get(code.upper())

    def search_by_description(self, keyword):
        """
        Search for ICD-10 codes by description keyword.

        Args:
            keyword (str): Keyword to search in descriptions

        Returns:
            list: List of matching codes and descriptions
        """
        keyword_lower = keyword.lower()
        results = []
        for code, data in self.codes_index.items():
            if keyword_lower in data["description"].lower():
                results.append(data)
        return results

    def get_category(self, category_name):
        """
        Get all codes in a specific category.

        Args:
            category_name (str): Name of the category

        Returns:
            list: List of codes in the category
        """
        return self.categories_index.get(category_name, [])

    def get_statistics(self):
        """
        Get statistics about the loaded ICD-10 data.

        Returns:
            dict: Statistics dictionary
        """
        return {
            "total_codes": len(self.codes_index),
            "total_categories": len(self.categories_index),
            "metadata": self.metadata,
        }
