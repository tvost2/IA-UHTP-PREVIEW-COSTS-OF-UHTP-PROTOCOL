import random

class ProtocoloIA:
    def __init__(self):
        self.nome = "NetMind-UHTP"
        self.versao = "0.1"
        self.cabecalho = {"protocol": "UHTP", "version": "0.1"}
        self.handshake = ["SYN", "ACK", "READY"]
        self.retry_limit = 3

    def handshake_automatizado(self):
        for tentativa in range(self.retry_limit):
            print(f"🔁 Handshake tentativa {tentativa+1}: {self.handshake}")
            sucesso = random.random() > 0.2
            if sucesso:
                return True
        return False

    def transmitir(self, dados):
        if self.handshake_automatizado():
            return f"{self.cabecalho}|DATA:{dados}"
        else:
            return "ERRO: Handshake falhou"
