import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
import logging
from typing import List, Dict, Tuple
from pathlib import Path

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constantes
HUES = {
    'pista': 115,
    'carro': 93,
    'cone': 23,
    'pessoa': 146,
    'gramado': 65,
    'obstaculo': 0
}
LABELS = sorted(HUES.keys())
DATA_DIR = Path('images')
ANNOT_DIR = Path('annotated')

class ImageProcessor:
    """Classe para processamento de imagens e anotações."""
    
    @staticmethod
    def load_image(image_path: Path, grayscale: bool = False) -> np.ndarray:
        """Carrega uma imagem em modo colorido ou escala de cinza."""
        try:
            mode = 0 if grayscale else 1
            img = cv2.imread(str(image_path), mode)
            if img is None:
                raise ValueError(f"Não foi possível carregar a imagem: {image_path}")
            return img
        except Exception as e:
            logger.error(f"Erro ao carregar imagem {image_path}: {str(e)}")
            raise

    @staticmethod
    def load_polygons(ann_path: Path) -> List[Dict]:
        """Carrega polígonos de um arquivo JSON de anotações."""
        try:
            with open(ann_path, 'r') as handle:
                data = json.load(handle)
            return data.get('shapes', [])
        except Exception as e:
            logger.error(f"Erro ao carregar anotações {ann_path}: {str(e)}")
            raise

    @staticmethod
    def plot_pair(images: List[np.ndarray], titles: List[str] = None, grayscale: bool = False) -> None:
        """Exibe duas imagens lado a lado."""
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        for i, (img, ax) in enumerate(zip(images, axes)):
            ax.imshow(img, cmap='gray' if grayscale else None)
            ax.axis('off')
            if titles and i < len(titles):
                ax.set_title(titles[i])
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_binary_mask(img_shape: Tuple[int, int], shape_dicts: List[Dict]) -> np.ndarray:
        """Cria máscara binária a partir de polígonos."""
        mask = np.zeros(img_shape[:2], dtype=np.float32)
        for shape in shape_dicts:
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        return mask

    @staticmethod
    def create_multi_mask(img_shape: Tuple[int, int], shape_dicts: List[Dict]) -> np.ndarray:
        """Cria máscaras multiclasse."""
        channels = []
        background = np.zeros(img_shape[:2], dtype=np.float32)
        label2poly = {s['label']: np.array(s['points'], dtype=np.int32) for s in shape_dicts}

        for label in LABELS:
            mask = np.zeros(img_shape[:2], dtype=np.float32)
            if label in label2poly:
                cv2.fillPoly(mask, [label2poly[label]], 255)
                cv2.fillPoly(background, [label2poly[label]], 255)
            channels.append(mask)

        # Adiciona máscara de fundo
        _, thresh = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
        channels.append(thresh)
        return np.stack(channels, axis=2)

    @staticmethod
    def draw_color_mask(img_shape: Tuple[int, int], shape_dicts: List[Dict]) -> np.ndarray:
        """Desenha máscaras coloridas usando cores HSV."""
        mask = np.zeros((*img_shape[:2], 3), dtype=np.uint8)
        label2poly = {s['label']: np.array(s['points'], dtype=np.int32) for s in shape_dicts}

        for label in LABELS:
            if label in label2poly:
                cv2.fillPoly(mask, [label2poly[label]], (HUES[label], 255, 255))
        
        return cv2.cvtColor(mask, cv2.COLOR_HSV2RGB)

def main():
    """Função principal para processar imagens e anotações."""
    processor = ImageProcessor()

    # Lista e ordena arquivos
    try:
        image_files = sorted(DATA_DIR.glob('*.png'), key=lambda x: int(x.stem))
        annot_files = sorted(ANNOT_DIR.glob('*.json'), key=lambda x: int(x.stem))
        
        if len(image_files) != len(annot_files):
            raise ValueError("Número de imagens e anotações não corresponde")
    except Exception as e:
        logger.error(f"Erro ao listar arquivos: {str(e)}")
        return

    # Processamento de máscaras binárias
    for img_path, ann_path in zip(image_files, annot_files):
        try:
            img = processor.load_image(img_path, grayscale=True)
            shapes = processor.load_polygons(ann_path)
            binary_mask = processor.create_binary_mask(img.shape, shapes)
            processor.plot_pair([img, binary_mask], ['Original', 'Máscara Binária'], grayscale=True)
        except Exception as e:
            logger.error(f"Erro ao processar {img_path}: {str(e)}")
            continue

    # Processamento de máscaras multiclasse (primeira imagem)
    if image_files and annot_files:
        try:
            img = processor.load_image(image_files[0], grayscale=True)
            shapes = processor.load_polygons(annot_files[0])
            multi_mask = processor.create_multi_mask(img.shape, shapes)
            
            # Exibe primeiras 5 máscaras
            for i in range(min(5, multi_mask.shape[2])):
                plt.imshow(multi_mask[:, :, i], cmap='gray')
                plt.title(f'Máscara: {LABELS[i] if i < len(LABELS) else "Fundo"}')
                plt.axis('off')
                plt.show()
        except Exception as e:
            logger.error(f"Erro ao processar máscaras multiclasse: {str(e)}")

    # Processamento de máscaras coloridas (até 5 imagens)
    for i, (img_path, ann_path) in enumerate(zip(image_files, annot_files)):
        if i >= 5:
            break
        try:
            img = processor.load_image(img_path)
            shapes = processor.load_polygons(ann_path)
            color_mask = processor.draw_color_mask(img.shape, shapes)
            Image.fromarray(color_mask).show()
        except Exception as e:
            logger.error(f"Erro ao processar máscara colorida {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    main()