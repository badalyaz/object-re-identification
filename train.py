import os
import torch
import string    
import random 
import argparse
import torchreid


class NewDataset(torchreid.data.datasets.ImageDataset):
    dataset_dir = ''

    def __init__(self, path, root='', **kwargs):
        self.train_dir = self.dataset_dir     
        self.query_dir = self.dataset_dir
        self.gallery_dir = self.dataset_dir
        
        train = self.process_dir(self.train_dir, isQuery=False)
        query = self.process_dir(self.query_dir, isQuery=True)
        gallery = self.process_dir(self.gallery_dir, isQuery=False)

        super(NewDataset, self).__init__(train, query, gallery, **kwargs)
        
        
    def process_dir(self, dir_path, isQuery, relabel=False):
        img_paths = glob(osp.join(dir_path, '*.jpg'))
        
        data = []
        for img_path in img_paths:

            img_name = img_path.split('/')[-1]
            name_splitted = img_name.split('_')
            pid = int( name_splitted[1][1:] )
            camid = int( name_splitted[0][1:] )

            if isQuery:
                camid += 10  # index starts from 0

            data.append((img_path, pid, camid))

        return data
    
    
    def process_dir_market(self, dir_path, relabel=False):
        img_paths = glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
    
    
def get_parser():
    parser = argparse.ArgumentParser()    

    parser.add_argument('--name', type=str, default='osnet_x1_0', help="ReID model name")
    parser.add_argument('--img_h', type=int, default=256, help="image height")
    parser.add_argument('--img_w', type=int, default=128, help="image width")
    parser.add_argument('--bs', type=int, default=32, help="batch size")
    parser.add_argument('--optim', type=str, default='adam', help="optimzer")
    parser.add_argument('--lr', type=float, default=0.003, help="learning rate")
    parser.add_argument('--lr_sch', type=str, default="single_step", help="learning rate scheduler")
    parser.add_argument('--step', type=int, default=5, help="learning rate scheduler's step size")
    parser.add_argument('--epochs', type=int, default=20, help="epoch count for the training loop")
    parser.add_argument('--eval_freq', type=int, default=5, help="evaluation frequency")
    parser.add_argument('--videos_paths', type=str, default='path/to/folder', help="video data folder path")
    parser.add_argument('--skip_frames', type=int, default=15, help="take every N-th frame from every video for data augmentation")
    parser.add_argument('--aug_count', type=int, default=5, help="number of augmentations to be applied on every image")
    parser.add_argument('--save_path', type=str, default='path/to/save', help="path to save data")
        
    args = parser.parse_args()
    
    return args


def augment_images(img, count):
    imgs = [img]

    for i in range(count):
        aug = iaa.Sequential([])

        rand_number = np.random.randint(0, 101)
        if rand_number < 33:
            aug.append(iaa.AdditiveGaussianNoise(loc=0, scale=(0.01*255, 0.08*255)))
        elif rand_number < 70:
            aug.append(iaa.AverageBlur(k=(3, 3)))

        rand_number = np.random.randint(0, 101)
        if rand_number < 30:
            aug.append(iaa.Multiply((0.7, 1.2)))
        elif rand_number < 70:
            aug.append(iaa.GammaContrast((1, 1.6)))            
            
        rand_number = np.random.randint(0, 101)
        if rand_number < 33:
            aug.append(iaa.ChangeColorTemperature((1100, 10000)))
        elif rand_number < 66:
            aug.append(iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True))           
                  
        rand_number = np.random.randint(0, 101)
        if rand_number < 33:
            aug.append(iaa.CoarseDropout(0.015, size_percent=0.1, per_channel=0.5))
        elif rand_number < 66:
            aug.append(iaa.SaltAndPepper(0.05, per_channel=True))
            
        rand_number = np.random.randint(0, 101)
        if rand_number < 50:
            aug.append(iaa.pillike.FilterEdgeEnhanceMore())

        aug.append(iaa.Fliplr(0.5))
        aug.append(iaa.AveragePooling((1, 3)))

        img_aug = aug(image=img)
        imgs.append(img_aug)

    return imgs


def create_data(args):
    pid = -1
    counter = 0

    for video_path in sorted(glob(args.videos_paths + '/*')):
        print(f'Preprocessing {video_path} video...')
        cap = cv2.VideoCapture(video_path)

        frame_counter = 0
        pid += 1
        while cap.isOpened():
            ret, frame = cap.read()
            frame_counter += 1

            if frame_counter % args.skip_frames == 0:
                if ret:
                    results = yolo_model(frame) 
                    result_pandas = results.pandas().xyxy[0]
                    people = result_pandas[result_pandas['name'] == 'person'][['xmin','ymin','xmax','ymax']]

                    if len(people) == 0:
                        continue

                    xyxy = people.to_numpy().astype(np.int32)[0]  # taking first person's bbox 
                    person = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]  # cropping the person from the frame
                    person = cv2.resize(person, (args.img_w, args.img_h))

                    images =  augment_images(person, count=args.aug_count)
                    for image in images:
                        counter += 1
                        name = f'c0_p{pid}_{counter}.jpg'
                        cv2.imwrite(f'{args.save_path}/{name}', image)
                else:
                    break
        print('Done!')

        
def main(args):
    create_data(args)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NewDataset.dataset_dir = args.save_path
    dataset_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=random.randint(1, 25)))
    torchreid.data.register_image_dataset(dataset_name, NewDataset)

    datamanager = torchreid.data.ImageDataManager(
        sources=dataset_name, 
        height=args.img_h, 
        width=args.img_w, 
        batch_size_train=args.bs, 
        batch_size_test=100,
        transforms=["random_flip", "random_crop"]
    )

    model = torchreid.models.build_model(
        name=args.name,
        num_classes=datamanager.num_train_pids,
        loss="triplet",
        pretrained=True
    ).to(device).train()


    optimizer = torchreid.optim.build_optimizer(
        model,
        optim=args.optim,
        lr=args.lr, 
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler=args.lr_sch, 
        stepsize=args.step,
    )

    engine = torchreid.engine.ImageTripletEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        margin=0.3,  # by default 0.3
        weight_t=1,  # weight for triplet loss
        weight_x=50, # weight for softmax loss
    )

    engine.run(
        save_dir=f"log/{args.name}",
        max_epoch=args.epochs, 
        eval_freq=args.eval_freq, 
        print_freq=50,
        test_only=False
    )

    
if __name__ == '__main__':
    args = get_parser()
    main(args)
