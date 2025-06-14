from PIL import Image
import io
import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.io import imsave
from config import model_name, logbase, imshape, labels, hues, n_classes
import shutil


class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        # Limpa o diretório de métricas ao iniciar
        tmp_metrics = os.path.join(logbase, 'metrics')
        if os.path.exists(tmp_metrics):
            shutil.rmtree(tmp_metrics)
            os.mkdir(tmp_metrics)

        # Configura o TensorBoard padrão para as métricas de treino
        # Ele loga em 'logbase/metrics/model_name/train' por padrão
        training_log_dir = os.path.join(logbase, 'metrics', model_name)
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Configura um writer separado para as métricas de validação
        self.val_log_dir = os.path.join(logbase, 'metrics', model_name + '_val')
        # CORREÇÃO CRÍTICA AQUI: Usando tf.compat.v1.summary.FileWriter
        self.val_writer = tf.compat.v1.summary.FileWriter(self.val_log_dir)


    def set_model(self, model):
        super(TrainValTensorBoard, self).set_model(model)


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}

        # Logar métricas de validação com o writer separado
        # REMOVIDO: 'with self.val_writer.as_default():' - Isso era o erro!
        for name, value in val_logs.items():
            # CORREÇÃO: Usando tf.compat.v1.Summary e add_summary direto no writer
            summary = tf.compat.v1.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch) # 'epoch' como step é crucial aqui
        self.val_writer.flush()

        # Passar as métricas de treino (e as val_ que não começam com 'val_') para o callback pai
        logs_for_super = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs_for_super)

    def on_train_end(self, logs=None):
        super().on_train_end(logs)
        self.val_writer.close()


class TensorBoardMask(tf.keras.callbacks.Callback):
    def __init__(self, log_freq):
        super().__init__()
        self.log_freq = log_freq
        self.im_summaries = []
        self.global_batch = -1 # Definimos como -1 inicialmente
        
        # Mudei o nome do diretório para evitar conflito com as imagens originais
        tmp_logdir = os.path.join(logbase, 'images') 
        if os.path.exists(tmp_logdir):
            shutil.rmtree(tmp_logdir)
            os.mkdir(tmp_logdir)
        self.logdir = tmp_logdir
        
        # CORREÇÃO CRÍTICA AQUI: Usando tf.compat.v1.summary.FileWriter
        self.writer = tf.compat.v1.summary.FileWriter(self.logdir)
        #print(f"DEBUG TB_MASK: Log directory for images set to: {self.logdir}")


    def _file_generator(self, path):
        if not os.path.exists(path):
            #print(f"DEBUG TB_MASK: Directory does not exist: {path}")
            return []
        files = [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
        return files


    def make_image(self, path):
        image = Image.open(path)
        output = io.BytesIO()
        image.save(output, format='PNG')
        image_string = output.getvalue()
        output.close()
        # CORREÇÃO AQUI: Usando tf.compat.v1.Summary.Image
        summary = tf.compat.v1.Summary.Image(height=imshape[0],
                                   width=imshape[1],
                                   colorspace=imshape[2],
                                   encoded_image_string=image_string)
        return summary


    def log_mask(self):
        # CAMINHO PARA AS IMAGENS ORIGINAIS DO SEU DATASET PARA PREVER
        images_to_predict_from_dir = os.path.join(logbase, 'images') # <--- AJUSTE ESTE CAMINHO SE NECESSÁRIO

        #print(f"\nDEBUG TB_MASK: Attempting to log masks for epoch {self.global_batch}.")
        #print(f"DEBUG TB_MASK: Looking for original images in: {images_to_predict_from_dir}")
        
        original_image_filenames = self._file_generator(images_to_predict_from_dir)
        if not original_image_filenames:
            #print(f"DEBUG TB_MASK: No original images found in {images_to_predict_from_dir}. Cannot log masks.")
            return

        count = 0
        for i, fn in enumerate(original_image_filenames):
            if count >= 5: # Limita a 5 máscaras por log
                print("DEBUG TB_MASK: Logged 5 masks for this epoch, skipping remaining.")
                break 

            im_path = os.path.join(images_to_predict_from_dir, fn)
            
            if not hasattr(self, 'model') or self.model is None:
                print(f"ERROR TB_MASK: Model not available in callback for prediction at epoch {self.global_batch}. Skipping image: {fn}")
                continue

            try:
                mask = self.predict(im_path)
                save_path = os.path.join(self.logdir, f'mask_epoch{self.global_batch}_sample_{i}.png') 
                imsave(save_path, mask)
                #print(f"DEBUG TB_MASK: Mask saved locally: {save_path}")

                # CORREÇÃO AQUI: Usando tf.compat.v1.Summary.Value
                image_summary = self.make_image(save_path)
                self.im_summaries.append(tf.compat.v1.Summary.Value(tag=f'predicted_mask/epoch_{self.global_batch}_sample_{i}', image=image_summary))
                count += 1
                #print(f"DEBUG TB_MASK: Mask summary appended for {fn}. Total summaries for this epoch: {len(self.im_summaries)}")

            except Exception as e:
                print(f"ERROR TB_MASK: Failed to process or save mask for {fn}: {e}")


    def add_masks(self, pred):
        blank = np.zeros(shape=imshape, dtype=np.uint8)

        for i, label in enumerate(labels):
            hue = np.full(shape=(imshape[0], imshape[1]), fill_value=hues[label], dtype=np.uint8)
            sat = np.full(shape=(imshape[0], imshape[1]), fill_value=255, dtype=np.uint8)
            val = pred[:,:,i].astype(np.uint8)

            im_hsv = cv2.merge([hue, sat, val])
            im_rgb = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB)
            blank = cv2.add(blank, im_rgb)

        return blank


    def predict(self, path):
        if imshape[2] == 1:
            im = cv2.imread(path, 0)
            im = im.reshape(im.shape[0], imshape[1], 1)
        elif imshape[2] == 3:
            im = cv2.imread(path, 1)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = im.reshape(im.shape[0], im.shape[1], 3)
        im = np.expand_dims(im, axis=0)
        
        pred = self.model.predict(im)
        pred = np.squeeze(pred) * 255.0
        if n_classes == 1:
            mask = np.array(pred, dtype=np.uint8)
        elif n_classes > 1:
            mask = self.add_masks(pred)
        return mask


    def write_summaries(self):
        if self.im_summaries:
            summary = tf.compat.v1.Summary(value=self.im_summaries)
            self.writer.add_summary(summary, self.global_batch) # 'global_batch' como step
            self.writer.flush() # Adicione flush para garantir que os dados sejam gravados
            ##print(f"DEBUG TB_MASK: Wrote {len(self.im_summaries)} image summaries for epoch {self.global_batch}.")
        else:
            pass
           # print(f"DEBUG TB_MASK: No image summaries to write for epoch {self.global_batch}.")
        self.im_summaries = [] # Limpa a lista para a próxima época


    def on_epoch_end(self, epoch, logs={}):
        # Retorna se não for múltiplo da frequência de log
        if int(epoch % self.log_freq) != 0:
            #print(f"DEBUG TB_MASK: Skipping image log for epoch {epoch} (log_freq={self.log_freq}).")
            return

        self.global_batch = epoch # Define o step para a época atual
        ##print(f"DEBUG TB_MASK: Logging images for epoch {epoch} (global_batch={self.global_batch}).")
        self.log_mask() # Prepara e adiciona máscaras a self.im_summaries
        self.write_summaries() # Escreve as máscaras no TensorBoard
        ##print(f"DEBUG TB_MASK: Finished processing images for epoch {epoch}.")
