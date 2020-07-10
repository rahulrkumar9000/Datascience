# -*- coding: utf-8 -*-
"""


@author: Rahul Kumar
"""

def output_table(Dir,IMG_Size=150,num_samples=5,label_0,label_1):
    z=[]
    IMG_SIZE=IMG_Size
    Dir=Dir
    for img in tqdm(os.listdir(Dir)):
        path = os.path.join(Dir,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        z.append(np.array(img))

    img_ids= pd.DataFrame(os.listdir(Dir),columns=["Image_id"])     
    pred_z= modelx.predict(np.array(z))
    pred_digits_z=np.argmax(pred,axis=1)
    preds_z=np.round(pred_z,0)
    categorical_preds_z = pd.DataFrame(preds_z).idxmax(axis=1)
    Pred_images= pd.concat([img_ids,categorical_preds_z],axis=1)
    Pred_images.columns=["Image_id","Predict"]
    Pred_images.loc[Pred_images['Predict'] == 0, 'Prediction'] = label_0 
    Pred_images.loc[Pred_images['Predict'] == 1, 'Prediction'] = label_1
    return display(HTML(Pred_images[["Image_id","Prediction"]].head(num_samples).to_html()))