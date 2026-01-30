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
    
    def process_grid_images(self, obs_list, settings):
        """
        un tensore a 4 dimensioni per agente 
        e una lista di dizionari
            - name: food
              index: 0
              color: [133, 188, 107] # green
            - name: agent
              index: 1
              color: [33, 150, 243] # blue
            - name: wall
              index: 2
              color: [100, 100, 100] # gray
            - name: bad food
              index: 3
              color: [190, 58, 39] # red
            - name: frozen agent
              index: 4
              color: [0, 255, 255] # light blue
        """
        self.step += 1
        result = []
        GRID_LOG_DIR = "logs_grid_images"
        if not os.path.exists(GRID_LOG_DIR):
            os.makedirs(GRID_LOG_DIR)

        height = settings['height']
        width = settings['width']
        bg_color = [0, 0, 0]

        for agent_id, obs in enumerate(obs_list):
            
            rgb_image = np.full((height, width, 3), bg_color, dtype=np.uint8)
            # singolo agente, n matrici quanti sono i tags
            for tag_tensor, tag_info in zip(obs, settings['tags']):
                color = tag_info['color']
                rgb_image[tag_tensor > 0.5] = color

            # adding agent at the center
            center_y, center_x = height // 2, width // 2
            rgb_image[center_y, center_x] = [0, 255, 0]

            img = PIL.Image.fromarray(rgb_image, mode='RGB')

            img = img.resize((256, 256), resample=PIL.Image.NEAREST)

            img_filename = f"agent-{agent_id}_{self.step}_grid.png"
            img.save(os.path.join(GRID_LOG_DIR, img_filename))

            buffered = io.BytesIO()
            img.save(buffered, format="PNG")

            result.append(base64.b64encode(buffered.getvalue()).decode('utf-8'))

        return result
        

