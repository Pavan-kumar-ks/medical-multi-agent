import obo_parser
import os

def parse_obo_to_rag_format(obo_file_path):
    """
    Parses a .obo file and converts it into a list of strings for RAG.
    Each string contains the disease name, definition, and synonyms.
    """
    if not os.path.exists(obo_file_path):
        raise FileNotFoundError(f"OBO file not found at: {obo_file_path}")

    docs = []
    with open(obo_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    terms = content.split('[Term]')
    for term_str in terms:
        if not term_str.strip():
            continue

        lines = term_str.strip().split('\n')
        term_data = {}
        for line in lines:
            if ': ' in line:
                key, value = line.split(': ', 1)
                if key in term_data:
                    if isinstance(term_data[key], list):
                        term_data[key].append(value)
                    else:
                        term_data[key] = [term_data[key], value]
                else:
                    term_data[key] = value
        
        if 'name' in term_data and 'def' in term_data:
            name = term_data['name']
            definition = term_data['def'].split('"')[1] # Extract text within quotes
            
            synonyms = []
            if 'synonym' in term_data:
                if isinstance(term_data['synonym'], list):
                    synonyms = [s.split('"')[1] for s in term_data['synonym']]
                else:
                    synonyms = [term_data['synonym'].split('"')[1]]
            
            doc_string = f"Disease: {name}. Definition: {definition}."
            if synonyms:
                doc_string += f" Synonyms: {', '.join(synonyms)}."
            
            docs.append(doc_string)
            
    return docs

if __name__ == '__main__':
    # This is an example of how to use the parser.
    # The main ingestion script will call this function.
    ontology_file = os.path.join('temp_ontology', 'src', 'ontology', 'doid.obo')
    if os.path.exists(ontology_file):
        rag_docs = parse_obo_to_rag_format(ontology_file)
        print(f"Successfully parsed {len(rag_docs)} documents.")
        # print(rag_docs[:5])
    else:
        print(f"Could not find ontology file at {ontology_file}")

