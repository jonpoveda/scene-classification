from database import DatabaseFiles as DB

db = DB('/home/jon/repos/mcv/m3/scene-classificator/data')

print('exist? {}'.format(db.dataset_exists('train')))
print('empty? {}'.format(db.dataset_is_empty('train')))
image_relative_path = 'train/tallbuilding/test1.jpg'

descriptor = [10, 30]
label = ['tall', 'verytall']
db.save_descriptor(image_relative_path, descriptor, label)
print(db.load_descriptor(image_relative_path))
