import os
import sys

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
            
            # Handle 'def' being a list or a string
            defs = term_data['def']
            if isinstance(defs, list):
                definition = defs[0].split('"')[1] # Take the first definition
            else:
                definition = defs.split('"')[1] # Extract text within quotes
            
            synonyms = []
            if 'synonym' in term_data:
                if isinstance(term_data['synonym'], list):
                    synonyms = [s.split('"')[1] for s in term_data['synonym'] if '"' in s]
                else:
                    if '"' in term_data['synonym']:
                        synonyms = [term_data['synonym'].split('"')[1]]
            
            doc_string = f"Disease: {name}. Definition: {definition}."
            if synonyms:
                doc_string += f" Synonyms: {', '.join(synonyms)}."
            
            docs.append(doc_string)
            
    return docs

if __name__ == '__main__':
    ontology_file = os.path.join('temp_ontology', 'src', 'ontology', 'doid.obo')
    output_file = os.path.join('app', 'data', 'knowledge_base', 'knowledge_base.txt')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if os.path.exists(ontology_file):
        rag_docs = parse_obo_to_rag_format(ontology_file)
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc in rag_docs:
                f.write(doc + '\n')
        print(f"Successfully converted ontology to {output_file}")
    else:
        print(f"Could not find ontology file at {ontology_file}")
