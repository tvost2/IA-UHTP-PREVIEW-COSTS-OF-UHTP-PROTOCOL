from core.ai.treinamento import (
    NetMindNN,
    CAMADAS,
    gerar_ips_fake,
    gerar_circuito,
    codificar_circuito,
    simular_custo,
    gerar_dataset,
)

import torch
import logging
def carregar_modelo(device, caminho_peso="netmind_model1.pt"):
    model = NetMindNN(CAMADAS).to(device)
    model.load_state_dict(torch.load(caminho_peso, map_location=device))
    model.eval()
    logging.info(f"✅ Modelo carregado de {caminho_peso}")
    return model

def gerar_varios_circuitos(quantidade=10):
    ips = gerar_ips_fake()
    return [gerar_circuito(ips) for _ in range(quantidade)], ips

def avaliar_circuitos(model, circuitos, ips, device):
    entradas = [codificar_circuito(c, ips) for c in circuitos]
    X = torch.tensor(entradas, dtype=torch.float32).to(device)
    with torch.no_grad():
        custos = model(X).cpu().numpy().flatten()
    return list(zip(circuitos, custos))

def mostrar_circuitos_e_custos(circuitos_custos):
    for i, (circuito, custo) in enumerate(circuitos_custos):
        logging.info(f"Circuito {i+1}: {circuito} | Custo previsto: {custo:.2f}")

def selecionar_melhor_circuito(circuitos_custos):
    melhor = min(circuitos_custos, key=lambda x: x[1])
    logging.info(f"🏆 Melhor circuito: {melhor[0]} | Custo: {melhor[1]:.2f}")
    return melhor

def salvar_circuito(circuito, nome_arquivo="melhor_circuito.txt"):
    with open(nome_arquivo, "w") as f:
        f.write("\n".join(circuito))
    logging.info(f"💾 Circuito salvo em {nome_arquivo}")

def carregar_circuito(nome_arquivo="melhor_circuito.txt"):
    with open(nome_arquivo, "r") as f:
        circuito = [linha.strip() for linha in f.readlines()]
    logging.info(f"📂 Circuito carregado de {nome_arquivo}")
    return circuito

def avaliar_circuito_individual(model, circuito, ips, device):
    entrada = torch.tensor([codificar_circuito(circuito, ips)], dtype=torch.float32).to(device)
    with torch.no_grad():
        custo = model(entrada).item()
    logging.info(f"Circuito: {circuito} | Custo previsto: {custo:.2f}")
    return custo

# Exemplo de uso integrado:
def pipeline_analise_circuitos(device, n_circuitos=10):
    model = carregar_modelo(device)
    circuitos, ips = gerar_varios_circuitos(n_circuitos)
    circuitos_custos = avaliar_circuitos(model, circuitos, ips, device)
    mostrar_circuitos_e_custos(circuitos_custos)
    melhor_circuito, melhor_custo = selecionar_melhor_circuito(circuitos_custos)
    salvar_circuito(melhor_circuito)
    return melhor_circuito, melhor_custo

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline_analise_circuitos(device, n_circuitos=10)