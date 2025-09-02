import uuid

def generate_uuid_string(length):
    uuid_string = str(uuid.uuid4()).replace('-', '')
    return uuid_string[:length]