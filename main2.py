import torch
import logging
import time
from core.protocolos.protocolo_base import ProtocoloIA
from core.cripto.cripto_auto import CriptoNetMind
from core.ai.treinamento import NetMindNN, gerar_dataset, CAMADAS

# Configuração de logging para melhor rastreabilidade
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def detectar_dispositivo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"📟 Dispositivo detectado: {device}")
    if device.type == "cuda":
        logging.info(f"🚀 Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        logging.info("🖥️ Usando CPU")
    return device

def treinar_e_salvar_modelo(device, epochs=100, salvar_cada=10):
    logging.info("🧠 Iniciando treinamento da IA de rede NetMind...")

    X, y = gerar_dataset(1000)
    X, y = X.to(device), y.to(device)
    model = NetMindNN(CAMADAS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    melhor_loss = float('inf')
    inicio = time.time()
    for epoch in range(epochs):
        model.train()
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < melhor_loss:
            melhor_loss = loss.item()
            torch.save(model.state_dict(), "netmind_model_best.pt")

        if epoch % salvar_cada == 0 or epoch == epochs - 1:
            logging.info(f"  📊 Epoch {epoch} | Loss: {loss.item():.6f}")
            torch.save(model.state_dict(), "netmind_model1.pt")

    tempo_total = time.time() - inicio
    logging.info(f"⏱️ Treinamento concluído em {tempo_total:.2f} segundos.")
    logging.info(f"💾 Melhor modelo salvo como netmind_model_best.pt")
    return model

def testar_protocolo():
    logging.info("📡 Testando protocolo NetMind-UHTP...")
    proto = ProtocoloIA()
    resposta = proto.transmitir("Teste de envio pelo protocolo IA")
    logging.info(f"📨 Protocolo resposta: {resposta}")

def testar_criptografia():
    logging.info("🔐 Testando criptografia avançada NetMind...")
    cripto = CriptoNetMind("2u389u2dmsaoi32138912¨$@#@#@39u28vu%$#@8921u4c921uc8432u3ck@#w#$kewke%¨¨qkwkeqwjeiqjweojqwoqjeoiqjiqwjeoqjeioj2icjei2j")
    mensagem = "Mensagem secreta via NetMind"
    cifrado = cripto.criptografar(mensagem)
    decifrado = cripto.descriptografar(cifrado)
    logging.info(f"🔏 Mensagem original: {mensagem}")
    logging.info(f"🔓 Mensagem decifrada: {decifrado}")

def main():
    device = detectar_dispositivo()
    model = treinar_e_salvar_modelo(device, epochs=10000000, salvar_cada=10)
    testar_protocolo()
    testar_criptografia()
    logging.info("🚀 Execução completa do NetMind integrada.")

if __name__ == "__main__":
    main()