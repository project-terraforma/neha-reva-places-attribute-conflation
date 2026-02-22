from deepparse.parser import AddressParser
import re

# Initialize the parser (this downloads a pre-trained model on first run)
address_parser = AddressParser(model_type="fasttext", device="cpu")

def get_address_score(address_str):
    """Parses address using DeepParse and returns a completeness score."""
    if not address_str or len(str(address_str)) < 5:
        return 0
    
    try:
        # DeepParse returns a parsed object
        parsed_address = address_parser(address_str)
        
        # Access components (StreetNumber, StreetName, Municipality, Province, PostalCode)
        components = parsed_address.to_dict()
        
        score = 0
        if components.get('StreetNumber'): score += 1
        if components.get('StreetName'): score += 1
        if components.get('Municipality'): score += 1 # City
        if components.get('Province'): score += 1     # State
        
        # Postcode validation
        postcode = components.get('PostalCode')
        if postcode:
            if re.match(r'^[A-Z0-9 -]{3,10}$', str(postcode).upper()):
                score += 1
                
        return score
    except Exception:
        return 0

def compare_addresses(base_addr, alt_addr):
    """Determines which address wins the +1 point for your Step 2 logic."""
    base_score = get_address_score(base_addr)
    alt_score = get_address_score(alt_addr)
    
    if base_score > alt_score:
        return "base"
    elif alt_score > base_score:
        return "alt"
    else:
        return None