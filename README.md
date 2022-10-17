# newpatch_rl

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>
  
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/shighghyujie/newpatch_rl/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
  
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->


```bash
$ git clone https://github.com/shighghyujie/newpatch_rl.git
$ cd newpatch_rl
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Preparation Before Attack</summary>
  
  The Opening Database LFW includes 13233 images, 5749 people, 1680 people with two or more images, which could be downloaded from [Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
  
  Firstly, you should put the folder "stmodels" in "newpatch_rl\code_rl\rlpatch".
  
  The folder is at: [Download from Baidu Drive](https://pan.baidu.com/s/1TKMwDJY6OnXPXzbnvet-PQ) ,code:q1w2
  
  You could skip the follow steps in preparation.or if you want to use your own database instead of LFW DataBase, you should prepare your own database. The database structure is as follows:
  
  the database should obey the Folder Structure:
  
 Directory structure:
 - DATASET_BASE
     - People's Name
         - img(jpg)
  
  Then you can execute the command as follow:
  
  ```bash
  $ cd code_rl/rlpatch
  $ python create_new_ens.py --database_path Your_Database_Path --new_add 0
  ```

</details>

<details open>
<summary>Attack</summary>
  
Firstly, you should prepare the folder of sacrificed faces
Directory structure:
- DATASET_BASE
    - People's Name
        - img(jpg)
  
if the sacrificed person wears the face mask,you should put another image without face mask before it. Just like this:
  
![Capture](source/1.png)

you can determine:
  
'targeted' 'True' to perform target attack or 'False' to perform nontarget attack,
  
'masked' 'True' to attack a sacrifice face with face mask or 'False' without face mask,
                  
'sticker_width' and "sticker_height" to change the size of adv patch,
  
'sacrificed_face_path' where sacrifice faces are from.
  
  
  
```bash
$ cd code_rl/rlpatch
$ python target_attack.py --targeted True --masked True --sticker_width [Width] --sticker_height [Height] --ens_num [num of your face base] --sacrificed_face_path [sarcificed face path]
```
 
for example:
  
you could perform the demo as follow:
  
```bash
$ python target_attack.py --targeted True --masked True  --sacrificed_face_path ../sacrifice_face_images
```
  
or you use your face database and you should offer the num of database, and the path of database:
  
```bash
$ python target_attack.py --targeted True --masked True --ens_num [num of your face base] --sacrificed_face_path [sarcificed face path]
 ```
