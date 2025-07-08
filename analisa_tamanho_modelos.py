import os

def tamanho_em_mb(caminho_arquivo):
    tamanho_bytes = os.path.getsize(caminho_arquivo)
    return tamanho_bytes / (1024 * 1024)  # converte para MB

def analisar_tamanhos(pasta_modelos):
    print(f"{'Arquivo':40s} | {'Tamanho (MB)':>12s}")
    print("-" * 56)
    for arquivo in os.listdir(pasta_modelos):
        caminho = os.path.join(pasta_modelos, arquivo)
        if os.path.isfile(caminho) and (arquivo.endswith(".h5") or arquivo.endswith(".tflite")):
            tamanho = tamanho_em_mb(caminho)
            print(f"{arquivo:40s} | {tamanho:12.3f}")

if __name__ == "__main__":
    pasta = "models"  # ajuste aqui sua pasta dos modelos
    analisar_tamanhos(pasta)
