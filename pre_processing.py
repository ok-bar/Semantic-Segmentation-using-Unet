def convert_image_rgb(data):
    imgs=[]
    for i in tqdm(data):
        img = cv2.imread(i,cv2.COLOR_BGR2RGB)
        del i
        imgs.append(img)
    return imgs
  
  
def split_input_mask(data):
  inputs=[]
  mask=[]
  for i in data:
      a=i[:,:256]
      inputs.append(a)
      b=i[:,256:]
      mask.append(b)
  return inputs,mask
