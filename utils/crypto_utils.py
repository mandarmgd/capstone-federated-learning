# utils/crypto_utils.py
from cryptography.fernet import Fernet
import pickle

def generate_key():
    """
    Run once to generate a shared key for server + clients.
    Example:
        from utils.crypto_utils import generate_key
        print(generate_key())
    """
    return Fernet.generate_key()

class CryptoBox:
    """
    Simple wrapper around Fernet for encrypting/decrypting Python objects.
    """
    def __init__(self, key: bytes):
        self.cipher = Fernet(key)

    def encrypt_obj(self, obj):
        data = pickle.dumps(obj)
        token = self.cipher.encrypt(data)
        return token

    def decrypt_obj(self, token):
        data = self.cipher.decrypt(token)
        obj = pickle.loads(data)
        return obj
