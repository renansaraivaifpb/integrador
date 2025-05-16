import cv2
import os

# Pasta de entrada e saída
pasta_entrada = 'renomear_imagens'
pasta_saida = 'images_redimensionadas'

# Criar pasta de saída se não existir
os.makedirs(pasta_saida, exist_ok=True)

# Listar todos os arquivos da pasta
for nome_arquivo in os.listdir(pasta_entrada):
    caminho_imagem = os.path.join(pasta_entrada, nome_arquivo)

    # Ler imagem
    imagem = cv2.imread(caminho_imagem)

    # Verifica se foi carregada corretamente
    if imagem is None:
        print(f"Erro ao carregar: {caminho_imagem}")
        continue

    # Redimensionarr
    imagem_redimensionada = cv2.resize(imagem, (640, 576))

    # Caminho de saída
    caminho_saida = os.path.join(pasta_saida, nome_arquivo)

    # Salvar imagem redimensionada
    cv2.imwrite(caminho_saida, imagem_redimensionada)

    print(f"Processada: {nome_arquivo}")
