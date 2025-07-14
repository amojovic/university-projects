from pathlib import Path  # Uvoz modula za rad sa putanjama
from PIL import Image  # Uvoz PIL biblioteke za obradu slika
import numpy as np  # Uvoz NumPy biblioteke za rad sa nizovima
import time  # Uvoz modula za rad sa vremenom
from scipy.ndimage import gaussian_filter  # Uvoz funkcije za gaussian blur iz SciPy
import shutil  # Uvoz za kopiranje fajlova
import threading  # Uvoz za rad sa nitima
import queue   # Uvoz za rad sa redovima
import json             
from multiprocessing import Pool, cpu_count
import sys
import os

task_threads = []
def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())
    print("\n\n")

# Funkcija za konverziju slike u grayscale, filter 1
def grayscale(image_array):
    red_channel = image_array[..., 0]  # Ekstrakcija crvenog kanala
    green_channel = image_array[..., 1]  # Ekstrakcija zelenog kanala
    blue_channel = image_array[..., 2]  # Ekstrakcija plavog kanala
    grayscale_image = (red_channel * 0.299 + green_channel * 0.587 + blue_channel * 0.114)
    return grayscale_image.astype(np.uint8)  # Vraća grayscale sliku kao uint8

# Funkcija za primenu gaussian blur filtera, filter 2
def gaussian_blur(image_array, sigma=3):
    red_channel = gaussian_filter(image_array[..., 0], sigma=sigma)
    green_channel = gaussian_filter(image_array[..., 1], sigma=sigma)
    blue_channel = gaussian_filter(image_array[..., 2], sigma=sigma)
    blurred_image = np.zeros_like(image_array)
    blurred_image[..., 0] = red_channel
    blurred_image[..., 1] = green_channel
    blurred_image[..., 2] = blue_channel

    if image_array.shape[-1] == 4:
        alpha_channel = image_array[..., 3]
        blurred_image[..., 3] = alpha_channel

    blurred_image = np.clip(blurred_image, 0, 255)
    return blurred_image.astype(np.uint8)

# Funkcija za podesavanje osvetljenja slike, filter 3
def adjust_brightness(image_array, factor=1.0):
    mean_intensity = np.mean(image_array, axis=(0, 1), keepdims=True)
    image_array = (image_array - mean_intensity) * factor + mean_intensity
    adjusted_image = np.where(image_array < 0, 0, image_array)
    adjusted_image = np.where(adjusted_image > 255, 255, adjusted_image)
    return adjusted_image.astype(np.uint8)

# Klasa za upravljanje slikama u registru
class RegistarSlika:
    def __init__(self, directory):
        self.directory = Path(directory)    # put do foldera slika
        self.images = []
        self.current_id = 1
        self.load_images()

    def load_images(self):
        for image_file in self.directory.glob("*"):                   # vracam bukvalno sve slike u folderu
            if image_file.is_file() and not image_file.name.startswith("processed_"):
                size_before = image_file.stat().st_size
                image_record = {
                    'id': self.current_id,
                    'file': image_file.name,
                    'task_id': None,
                    'marked_for_deletion': False,
                    'filter_1_used': False,
                    'filter_2_used': False,     # mozda nije potrebno jer imamo listu zadataka ali neka se nadje
                    'filter_3_used': False,
                    'tasks': [],                # ovde bi bilo dobro beleziti zadatke koji su izvrseni nad slikom
                    'processing_time': None,
                    'size_before': size_before,
                    'size_after': size_before,   # ovo cu da menjam kada se izvrsi konkretna obrada slike i samim tim promena velicine
                }
                self.images.append(image_record)  # dodaj sliku u listu
                self.current_id += 1              # inkrementiraj za sledeci id   

    def add_image(self, image_path):
        image_file = Path(image_path)   # objekat putanje do slike gde se sada nalazi negde na disku
        if not image_file.is_file():
            print(f"Slika na putanji {image_path} ne postoji.")
            return None

        destination_path = self.directory / image_file.name   # destination je gde zelimo da stavimo sliku tj. slike folder + ime ovde nove slike appendujemo
        if not destination_path.exists():                     # ako nema na destination-u slika onda
            shutil.copy(image_file, destination_path)         # kopiramo sliku tamo
            print(f"Slika {image_file.name} prekopirana u {self.directory}.")

        size_before = destination_path.stat().st_size    # namestamo sve za sliku
        image_record = {
            'id': self.current_id,
            'file': image_file.name,
            'task_id': None,
            'marked_for_deletion': False,
            'filter_1_used': False,
            'filter_2_used': False,
            'filter_3_used': False,
            'tasks': [],
            'processing_time': None,
            'size_before': size_before,
            'size_after': size_before,
        }
        self.images.append(image_record)       # dodamo je u registar
        self.current_id += 1
        print(f"Slika {image_file.name} dodata u registar.")
        return image_record['id']  # Vrati id te slike

    def print_images(self):  # Ispis svih slika iz registra
        for image in self.images:
            print(f"Image ID: {image['id']}, File: {image['file']}, "
                  f"Task ID: {image['task_id']}, Size Before: {image['size_before']}, Size After: {image['size_after']}")
    
    def id_from_path(self, path):
        for image in self.images:
            if image['file'] == path.split("/")[-1]:
                return image['id']
    
    def set_task_id(self, image_id, task_id):
        for image in self.images:
            if image['id'] == image_id:
                image['task_id'] = task_id
                return True
        return False


# Klasa za upravljanje zadacima obrade slika
class RegistarZadataka:
    def __init__(self):
        self.tasks = []             # Lista zadataka
        self.current_task_id = 1    # Pri inicilizaciji prvi task setujem da id bude 1
        self.condition = threading.Condition()  # Uslov za zavrsetak zadatka

    def add_task(self, task_type):  # Funkcija koja dodaje zadatak u listu zadataka u registru na osnovu tipa koji prosledimo
        if task_type not in ['add_image', 'process','delete_image']:    # Moguci tipovi zadataka
            print("Nije validan tip zadatka.")
            return None
        
        # Struktura zadatka, za pri dodavanju setujemo id, tip. Kasnije cemo update status na nesto drugo i slika id na koji se odnosi
        task_record = {
            'task_type': task_type,
            'task_id': self.current_task_id,
            'filter_type': None,
            'input_image_path': None,
            'output_image_path': None,
            'status': 'cekanje',
            'image_id': None 
        }
        self.tasks.append(task_record)     # Dodamo zadatak u listu
        self.current_task_id += 1          # Inkrementiram id za sledeci zadatak
        return task_record['task_id']

    def update_task_status(self, task_id, new_status):         # Da menjamo status kako ide tok obrade
        valid_statuses = ['cekanje', 'u obradi', 'zavrseno']
        if new_status not in valid_statuses:
            print("Nije validan status.")
            return False
        
        with self.condition:  # Zakljucak pomocu contidion-a
            for task in self.tasks:
                if task['task_id'] == task_id:
                    if(task['status'] != new_status):
                        print(f"Promena statusa za task {task['task_id']} iz {task['status']} na {new_status}")
                    task['status'] = new_status   # namestimo novi status
                    if new_status == 'zavrseno':
                        self.condition.notify_all()  # Notify sve niti
                    return True
        return False

        
    def wait_for_task_completion(self, task_id):
        with self.condition:
            while not any(task['task_id'] == task_id and task['status'] == 'zavrseno' for task in self.tasks):
                self.condition.wait()  # cekaj notifikaciju da je zadatak zavrsen

    def set_image_id(self, task_id, image_id):  # Ovo koristim da namestim vezu izmedju odredjenog zadatka i slike
        for task in self.tasks:
            if task['task_id'] == task_id:
                task['image_id'] = image_id  
                return True
        return False

    def set_input_image_path(self, task_id, input_image_path):
        for task in self.tasks:
            if task['task_id'] == task_id:
                task['input_image_path'] = input_image_path 
                return True
        return False

    def set_output_image_path(self, task_id, output_image_path):
        for task in self.tasks:
            if task['task_id'] == task_id:
                task['output_image_path'] = output_image_path
                return True
        return False
    def set_filter_type(self, task_id, filter_type):
        for task in self.tasks:
            if task['task_id'] == task_id:
                task['filter_type'] = filter_type
                return True
        return False

    def print_tasks(self):
        for task in self.tasks:
            print(f"Task ID: {task['task_id']}, Image ID: {task['image_id']}, "
                  f"Task Type: {task['task_type']}, Status: {task['status']}")

registar_slika = RegistarSlika('./slike')  # Pravi registar zadataka (na osnovu foldera slike koji se podrazumeva da se nalazi u istom folderu kao ovaj kod.py
registar_zadataka = RegistarZadataka()  # Pravi registar zadataka


# Funkcija za ispis poruke
def print_message(message):
    print(f"Output: {message}")

# Funkcija za obradu komandi unetih od strane korisnika
def process_command(command, message_queue, exit_queue, running):
    parts = command.split()
    if len(parts) == 0:      # Ako nista nije uneto, samo nastavi dalje sa main loop-om
        return True

    cmd = parts[0]           # Uzimamo prvi deo komande

    if cmd == "add" and len(parts) == 2:    # Ako je u pitanju add to znaci da je trebalo da bude 2 dela, znaci add i putanje do slike
        image_path = parts[1]               # Pokupimo putanju slike
        task_id = registar_zadataka.add_task('add_image')  # U registar zadataka zavalimo novi zadatak tipa 'add_image' a ta funkcija nam vraca task_id iz registra
        if task_id:
            thread = threading.Thread(target=add_image_thread, args=(image_path, task_id)).start() # Pravimo novu nit gde cemo dodati sliku u folder(koristeci add_image_thead funkciju), startujemo tu nit
            task_threads.append(thread)

    elif cmd == "typeout" and len(parts) == 2:   # Ovo sam samo napravio da bi proverio ispis paralelno u konzolu, mozda kasnije obisem.
        message = parts[1]
        thread = threading.Thread(target=print_message, args=(message,)).start()
        task_threads.append(thread)
        
    elif cmd == "list":  # list komanda
        thread = threading.Thread(target=list_images_thread, args=(message_queue, )).start()
        task_threads.append(thread)
        
    elif cmd == "describe" and len(parts) == 2:  # describe komanda
        try:
            image_id = int(parts[1])  # hvata image id
            thread = threading.Thread(target=describe_image_thread, args=(image_id, message_queue)).start()
            task_threads.append(thread)
        except ValueError:
            print("Id nije validan.")
            
    elif cmd == "process" and len(parts) == 2:  
        json_file = parts[1]
        thread = threading.Thread(target=process_json, args=(json_file, )).start()
        task_threads.append(thread)
        
    elif cmd == "delete" and len(parts) == 2:
        filename = parts[1]
        for image in registar_slika.images:
            if image['file'] == filename:
                task_id = registar_zadataka.add_task('delete_image') 
                if task_id:
                    thread = threading.Thread(target=delete_image_thread, args=(filename, task_id)).start()
                    task_threads.append(thread)

    elif cmd == "exit":
        print("Exiting the system...")
        threading.Thread(target=exit_thread_func, args=(exit_queue, )).start()
        return False  

    else:
        print(f"Nepoznata komanda: {cmd}")   # Ako je uneto nesto sto nema smisla

    return True


def multiprocessing_callback_function(arg_tuple):
    #((id, path, size), (id, path, size), (id, path, size))
    for img_tuple in arg_tuple:
        task_id = img_tuple[0]
        img_path = img_tuple[1]
        old_size = img_tuple[2]
        registar_zadataka.update_task_status(task_id, "zavrseno")
        img_id = registar_slika.add_image(img_path)
        for image in registar_slika.images:
            if image['id'] == img_id:
                image['task_id'] = task_id
                image['size_before'] = old_size


def process_json(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error reading JSON file {json_file}: {e}")
        return

    for task_data in data.get('tasks', []):
        input_image_path = task_data.get('input_image_path')
        image_id = registar_slika.id_from_path(input_image_path)
        output_image_path = task_data.get('output_image_path')
        filter_type = task_data.get('filter_type')

        if not all([image_id, input_image_path, output_image_path, filter_type]):
            print(f"Missing data for task: {task_data}")
            continue
        
        task_id = registar_zadataka.add_task(task_type="process") 
        if task_id:
            registar_zadataka.set_image_id(task_id, image_id)
            registar_zadataka.set_input_image_path(task_id, input_image_path)
            registar_zadataka.set_output_image_path(task_id, output_image_path)
            registar_zadataka.set_filter_type(task_id, filter_type) 
            registar_zadataka.update_task_status(task_id, 'cekanje')
    
    tasklist = []
    for task in registar_zadataka.tasks:
        if task['status'] != 'zavrseno' and task['task_type'] == 'process':
            tasklist.append(task)
    
    with Pool(processes=cpu_count()) as pool:
        print(f"we mapping, {len(tasklist)}")
        res = pool.map_async(process_task, tasklist, callback=multiprocessing_callback_function, error_callback=lambda error: print(f"Error:\n{error}"))
        res.get()


def process_task(task):
    task_id = task['task_id']
    image_id = task['image_id']
    filter_type = task['filter_type']
    image_record = next((img for img in registar_slika.images if img['id'] == image_id), None)

    if not image_record:
        print(f"Image ID {image_id} not found!")
        return ("nothing", "None")
    input_image_path = registar_slika.directory / image_record['file']
    
    if not input_image_path.is_file():
        print(f"Input image {input_image_path} not found!")
        return ("nothing", "None")

    image = np.array(Image.open(input_image_path))
    if filter_type == 'filter_1':
        filtered_image = grayscale(image)
    elif filter_type == 'filter_2':
        filtered_image = gaussian_blur(image)
    elif filter_type == 'filter_3':
        filtered_image = adjust_brightness(image)
    else:
        print(f"Unknown filter type {filter_type}!")
        return ("nothing", "None")

    output_image_path = Path(task.get('output_image_path'))
    Image.fromarray(filtered_image).save(output_image_path)

    return (task_id, output_image_path, image_record['size_after'])


# Funkcija za dodavanje nove slike u registar
def add_image_thread(image_path, task_id):
    registar_zadataka.update_task_status(task_id, 'u obradi')  # Poziva da se update-uje status obrade"
    image_id = registar_slika.add_image(image_path)
    if image_id:  # Ako uspe dodavanje
        registar_zadataka.update_task_status(task_id, 'zavrseno')  # Dodata tako da zavrseno
        registar_zadataka.set_image_id(task_id, image_id)  # set id slike za u zadatku
        registar_slika.set_task_id(image_id, task_id)
    else:
        registar_zadataka.update_task_status(task_id, 'zavrseno')  # Iako nije dodata oznacimo ga zavrseni

def list_images_thread(message_queue):
    images_list = []                                  # znaci pravimo listu slika
    for image in registar_slika.images:
        full_path = registar_slika.directory / image['file']  # pravimo putanju do slike
        images_list.append(f"Image ID: {image['id']}, Path: {full_path}")   # od svake iz registra uzmemo id i putanju i dodamo u ovu novu listu
    message_queue.put("\n".join(images_list))  # Zavalim sve to u red

def describe_image_thread(image_id, message_queue):
    image_record = next((img for img in registar_slika.images if img['id'] == image_id), None)
    
    if image_record is None:
        message_queue.put(f"Slika sa id {image_id} ne postoji.")
        return

    # Uzmemo sve zadatke iz list zadataka u registru slika za ovu sliku
    associated_tasks = [task for task in registar_zadataka.tasks if task['image_id'] == image_id]
    # Spremamo poruku koju cemo da stavimo u red, znaci za svaki zadatak uzmemo type i ondah i stavljamo sve u listu
    task_descriptions = [f"Task ID: {task['task_id']}, Type: {task['task_type']}, Status: {task['status']}" for task in associated_tasks]
    task_list = "\n".join(task_descriptions) if task_descriptions else "No associated tasks."

    message = (
        f"Image ID: {image_record['id']}\n"
        f"File: {image_record['file']}\n"
        f"Zadaci:\n"
        f"{task_list}"
    )
    message_queue.put(message)

    

def delete_image_thread(filename, task_id):
    image_record = next((img for img in registar_slika.images if img['file'] == filename), None)
    
    registar_zadataka.update_task_status(task_id, 'u obradi') 

    if image_record is None:
        print(f"Image with filename '{filename}' not found in the registry.")
        registar_zadataka.update_task_status(task_id, 'zavrseno')
        return

    image_record['marked_for_deletion'] = True

    with registar_zadataka.condition:
        while any(task['image_id'] == image_record['id'] and task['status'] != 'zavrseno' for task in registar_zadataka.tasks):
            registar_zadataka.condition.wait()

    try:
        image_path = registar_slika.directory / image_record['file']
        if image_path.exists():
            image_path.unlink() 
            print(f"Deleted image file: {image_record['file']}")

        registar_slika.images.remove(image_record)
        print(f"Image '{filename}' removed from registry.")

        registar_zadataka.update_task_status(task_id, 'zavrseno')

    except Exception as e:
        print(f"Error deleting image '{filename}': {e}")
        registar_zadataka.update_task_status(task_id, 'zavrseno')
    
def exit_thread_func(exit_queue):
    
    for task in registar_zadataka.tasks:
        exit_queue.put(task)
    
    while not exit_queue.empty():
        task = exit_queue.get()
        registar_zadataka.wait_for_task_completion(task['task_id'])  
        
        exit_queue.task_done()
    
    print("All tasks completed. Closing threads...")
    for thread in task_threads:
        if thread is not None and hasattr(thread, 'join'):
            thread.join()
    
    print("Exit process complete.")
    sys.exit()

# Glavna nit
def main():
    message_queue = queue.Queue()              # Inicijalizuj red za poruke
    exit_queue = queue.Queue()
    
    print("Unesite komandu:") 
    running = True   #flag koji koristimo da regulisemo da li main nit treba da runnuje

    while running:  
        command = input("> ")  # Ceka na unos komande
        running = process_command(command, message_queue, exit_queue, running)  # Obrada komande + ako vrati false loop se prekida
        
         # Periodicna provera za red kao sto se trazi u specifikaciji
        while not message_queue.empty():  # Ako ima poruke
            message = message_queue.get()  # Uzmi poruku
            print(message)  # Ispisi poruku

    print("Tasks in the registry:")  # Ispisuje sve zadatke u registru
    registar_zadataka.print_tasks()  # Ispisuje sve zadatke
    print("Images in the registry:")  # Ispisuje sve slike u registru
    registar_slika.print_images()  # Ispisuje sve slike

# Ovo nam omogucava da se main() pozove samo kada kada kod.py direktno runnujemo. Inace bi bio pozvan ako bi ga neki drugi .py importovao. Ovako bi u tom drugom .py ustvari _name_ setovan na 'kod' i samim tim main() ne bi bio pozvan 
if __name__ == '__main__':
    main()  # Poziva glavnu funkciju