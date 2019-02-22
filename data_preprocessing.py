import os, shutil

# declare the paths of downloaded folder; Training, Validation and Test folder
downloaded_dir = '/Users/bathuy/Downloads/fruits-360'
valid_dir = os.path.join(downloaded_dir, 'Validation')
training_dir = os.path.join(downloaded_dir, 'Training')
test_dir = os.path.join(downloaded_dir, 'Test')

# create the validation folder, which is the subfolder of original downloaded folder
try:  # to check whether the folder already exists or not
    os.mkdir(valid_dir)
except:
    print("Validation folder exists")

# list all the folders inside Test folder
folders = os.listdir(test_dir)


# go through all these folders and copy half of images in each folder to make a validation set
for folder in folders:
    if folder != '.DS_Store':
        try: # to check whether the folder already exists or not
            os.mkdir(os.path.join(valid_dir, folder))
            dst = os.path.join(valid_dir, folder)
            files = os.listdir(os.path.join(test_dir, folder))
            fnames = [files[i] for i in range(len(files)//2)]
            for fname in fnames:
                src = os.path.join(test_dir, folder, fname)
                dst = os.path.join(valid_dir, folder, fname)
                shutil.move(src, dst)
        except:
            pass
