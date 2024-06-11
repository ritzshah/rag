import pandas as pd
from pathlib import Path

def update_pdf_metadata(pdf_docs, csv_path):
    csv_data = pd.read_csv(csv_path)
    
    # Print the column names to debug
    print("Column names in the CSV file:", csv_data.columns.tolist())

    path_column = 'PATH'
    url_column = 'JCR:CONTENT/METADATA/XMP:RHCC-SOURCE-URL'
    csv_data[path_column] = csv_data[path_column].astype(str)
    print(csv_data[path_column])
    csv_data.dropna(subset=[path_column, url_column], inplace=True)
    pdfs_to_urls = dict(zip(csv_data[path_column].apply(lambda x: Path(x).stem), csv_data[url_column]))

    for doc in pdf_docs:
        source_stem = Path(doc.metadata["source"]).stem
        if source_stem in pdfs_to_urls:
            doc.metadata["source"] = pdfs_to_urls[source_stem]
            print("Updated metadata for:", source_stem)
        else:
            doc.metadata["source"] = source_stem
            print("Using PATH as source for:", source_stem)

    print("Metadata update completed.")

