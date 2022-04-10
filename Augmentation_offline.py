
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import albumentations as A

def augmentation_offline(img_path, mask_path, img_path_n, mask_path_n, i, trans):
    """create augm. img and mask and save it
    """
    
    img = mpimg.imread(img_path)
    mask = load_img(mask_path, color_mode="grayscale")
    mask = img_to_array(mask)
    
    img_new_name = img_path.split('\\')[1].split('.')[0] + '_M{}.png'.format(i)
    img_new_name = img_path_n + "\\" + img_new_name
    
    mask_new_name = mask_path.split('\\')[1].split('.')[0] + '_M{}.png'.format(i)
    mask_new_name = mask_path_n + "\\" + mask_new_name
    
    if (trans == "hor_flip"):
        transform = A.Compose([A.HorizontalFlip(p=1)],
                            additional_targets={'mask': 'mask'})
        
    elif (trans == "ver_flip"):
        transform = A.Compose([A.VerticalFlip(p=1)],
                                additional_targets={'mask': 'mask'})
    elif (trans == "to_gray"):
        transform = A.Compose([A.ToGray(p=1)])
        
    elif (trans == "dropout"):
        transform = A.Compose([A.Cutout(num_holes=50, 
                                        max_h_size=20, 
                                        max_w_size=20, 
                                        fill_value=1, p=1)])
    
    elif (trans == "channel_dropout"):
        transform = A.Compose([A.ChannelDropout(p=1)])
        
    elif (trans == "rotate"):
        transform = A.Compose([A.ShiftScaleRotate(p=1)],
                                additional_targets={'mask': 'mask'})
    
    elif (trans == "blur"):
        transform = A.Compose([A.Blur(blur_limit=(5, 5), p=1)])
        
    elif (trans == "noice"):
        transform = A.Compose([A.MultiplicativeNoise(multiplier=[0.3, 1.5], 
                                                 elementwise=True, 
                                                 per_channel=True, p=1)])
    
    elif (trans == "elastic_transform"):
        transform = A.Compose([A.ElasticTransform(p=1, 
                                              alpha=120, 
                                              sigma=120 * 0.05,
                                              alpha_affine=120 * 0.03)
        ],
        additional_targets={'mask': 'mask'}
        )
    
    transfor_out = transform(image=img, mask=mask)
    transfor_out['mask'] = transfor_out['mask'].squeeze(axis=2)
    
    plt.imsave(img_new_name, transfor_out['image'])
    plt.imsave(mask_new_name, transfor_out['mask'])
    
if __name__ == '__main__':

    # offline data augmentation

    # new files
    img_mod = "img_mod"
    mask_mod = "mask_mod"


    # for i, path in enumerate(img_all):
        
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 1, 
    #                         'hor_flip')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 2, 
    #                         'ver_flip')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 3, 
    #                         'to_gray')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 4, 
    #                         'dropout')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 5, 
    #                         'channel_dropout')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 6, 
    #                         'rotate')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 7, 'blur')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 8, 'noice')
    #     augmentation_offline(path, mask_all[i], img_mod, mask_mod, 9, 
    #                         'elastic_transform')