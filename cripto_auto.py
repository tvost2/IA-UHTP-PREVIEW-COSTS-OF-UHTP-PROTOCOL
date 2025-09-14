from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import hashlib

class CriptoNetMind:
    def __init__(self, chave_base):
        self.chave_original = hashlib.sha256(chave_base.encode()).digest()

    def mutar_chave(self):
        return hashlib.sha256(self.chave_original[::-1]).digest()

    def criptografar(self, dados):
        chave = self.mutar_chave()
        nonce = get_random_bytes(12)
        cipher = AES.new(chave, AES.MODE_GCM, nonce=nonce)
        cifrado, tag = cipher.encrypt_and_digest(dados.encode())
        return nonce + tag + cifrado

    def descriptografar(self, pacote):
        chave = self.mutar_chave()
        nonce, tag, cifrado = pacote[:12], pacote[12:28], pacote[28:]
        cipher = AES.new(chave, AES.MODE_GCM, nonce=nonce)
        return cipher.decrypt_and_verify(cifrado, tag).decode()
