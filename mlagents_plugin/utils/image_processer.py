import os
import io
import base64
import numpy as np
import PIL.Image

class ImageProcesser:

    def __init__(self):
        self.step = 0

    def process_and_save_img(self, obs_array, agent_id):
        """
        Converte l'array di Unity in immagine, la salva su disco 
        e la prepara in Base64 per l'LLM.
        """
        IMAGE_LOG_DIR = "logs_immagini"
        if not os.path.exists(IMAGE_LOG_DIR):
            os.makedirs(IMAGE_LOG_DIR)

        img_array = np.squeeze(obs_array)
        
        if img_array.shape[0] == 3 or img_array.shape[0] == 1:
            img_array = np.transpose(img_array, (1, 2, 0))
        
        if img_array.max() <= 1.0:
            img_array = (img_array * 255).astype(np.uint8)
        
        mode = 'RGB' if img_array.shape[-1] == 3 else 'L'
        img = PIL.Image.fromarray(img_array, mode)
        
        img_resized = img.resize((256, 256), PIL.Image.NEAREST)
        
        img_filename = f"agent-{agent_id}_{self.step}.png"
        img_resized.save(os.path.join(IMAGE_LOG_DIR, img_filename))
        
        buffered = io.BytesIO()
        img_resized.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def process_batch_images(self, obs_list):
        """
        Prende l'intera lista di tensori e restituisce una lista di stringhe Base64.
        """
        self.step += 1
        result = [ self.process_and_save_img(obs, agent_id=i) for i, obs in enumerate(obs_list) ]
        return result

