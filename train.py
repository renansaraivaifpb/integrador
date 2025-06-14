import os
from tensorflow.keras.callbacks import ModelCheckpoint
from data_generator import DataGenerator
from tensorboard_callbacks import TrainValTensorBoard, TensorBoardMask
from utils import generate_missing_json
from config import model_name, n_classes
from models import unet, fcn_8
from colorama import Fore, Style, init
from tensorflow.keras.utils import plot_model
os.environ["PATH"] += os.pathsep + 'C:/Users/arqis/Documents/renan/Graphviz-12.2.1-win64/bin'


# Inicializa colorama para colorir os textos no terminal, com reset automático após cada print
init(autoreset=True)

def get_sorted_file_paths(directory: str) -> list:
    """Retorna uma lista ordenada de caminhos de arquivos numerados.

    Ordena arquivos pelo número do nome antes da extensão,
    garantindo que 1.png venha antes de 10.png, por exemplo.
    """
    filenames = sorted(os.listdir(directory), key=lambda x: int(os.path.splitext(x)[0]))
    return [os.path.join(directory, fname) for fname in filenames]

def prepare_data(images_dir='images', annots_dir='annotated'):
    """Verifica consistência dos dados e prepara os caminhos dos arquivos.

    Se a quantidade de imagens e anotações for diferente,
    gera os JSONs ausentes para as anotações.
    Retorna listas ordenadas de caminhos de imagens e anotações.
    """
    if len(os.listdir(images_dir)) != len(os.listdir(annots_dir)):
        generate_missing_json()
    image_paths = get_sorted_file_paths(images_dir)
    annot_paths = get_sorted_file_paths(annots_dir)
    return image_paths, annot_paths

def get_model(name: str):
    """Seleciona e retorna o modelo baseado no nome da configuração.

    Atualmente suporta 'unet' e 'fcn_8'.
    """
    if 'unet' in name:
        return unet(pretrained=False, base=4)
    elif 'fcn_8' in name:
        return fcn_8(pretrained=False, base=4)
    else:
        raise ValueError(f"Modelo '{name}' não suportado.")

def create_callbacks(model_name: str):
    # Cria a pasta 'models' caso não exista para salvar checkpoints e modelo final
    os.makedirs('models', exist_ok=True)

    checkpoint_path = os.path.join('models', f'{model_name}.h')
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,    # caminho do arquivo de checkpoint
        monitor='loss',              # monitorar a perda para salvar o melhor modelo
        verbose=1,                   # mostrar mensagens quando salvar
        mode='min',                  # salvar quando a perda diminuir
        save_best_only=True,         # salva somente o melhor modelo (menor loss)
        save_weights_only=False,     # salva o modelo completo, não só pesos
        save_freq='epoch'            # salva ao final de cada época
    )
    # Retorna lista de callbacks: checkpoint e os callbacks customizados para tensorboard
    return [checkpoint, TrainValTensorBoard(write_graph=True), TensorBoardMask(log_freq=10)]

def styled_model_summary(model):
    """Imprime o resumo do modelo com estilo colorido no terminal.

    Mostra nome, tipo, shape de saída e parâmetros de cada camada,
    e o total de parâmetros do modelo.
    """
    print(f"\n{Style.BRIGHT}{Fore.CYAN}📦 Model Summary: {model.name}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 100}{Style.RESET_ALL}")
    print(f"{Style.BRIGHT}{'Layer Name':<25} {'Type':<20} {'Output Shape':<30} {'Params':<10}")
    print(f"{Fore.YELLOW}{'─' * 100}{Style.RESET_ALL}")

    for layer in model.layers:
        name = f"{Fore.BLUE}{layer.name:<25}{Style.RESET_ALL}"
        ltype = f"{Fore.MAGENTA}{layer.__class__.__name__:<20}{Style.RESET_ALL}"
        out_shape = f"{Fore.WHITE}{str(layer.output_shape):<30}{Style.RESET_ALL}"
        params = f"{Fore.GREEN}{layer.count_params():<10,}{Style.RESET_ALL}"
        print(f"{name} {ltype} {out_shape} {params}")

    print(f"{Fore.YELLOW}{'─' * 100}")
    print(f"{Style.BRIGHT}{Fore.GREEN}🧠 Total Parameters: {model.count_params():,}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}{'─' * 100}\n")

def main():
    # Prepara os dados: listas ordenadas de imagens e anotações
    image_paths, annot_paths = prepare_data()

    # Obtém o modelo selecionado na configuração
    model = get_model(model_name)

    # Mostra o resumo do modelo de forma estilizada no terminal
    styled_model_summary(model)

    # Instancia o gerador de dados customizado para alimentar o modelo durante o treino
    tg = DataGenerator(
        image_paths=image_paths,
        annot_paths=annot_paths,
        batch_size=1,
        augment=True
    )

    # Cria os callbacks incluindo checkpoint e tensorboard
    callbacks = create_callbacks(model_name)

    # Treina o modelo com os dados gerados, número de épocas definido, verbose ligado e callbacks configurados
    model.fit(
        tg,
        steps_per_epoch=len(tg),
        epochs=300,
        verbose=1,
        callbacks=callbacks
    )


if __name__ == '__main__':
    main()
