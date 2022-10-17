# newpatch_rl

## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>
  
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/shighghyujie/newpatch-rl/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
  
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->


```bash
$ git clone https://github.com/shighghyujie/newpatch-rl.git
$ cd newpatch_rl
$ pip install -r requirements.txt
```

</details>

<details open>
<summary>Preparation Before Attack</summary>
  
  The Opening Database LFW includes 13233 images, 5749 people, 1680 people with two or more images, which could be downloaded from [Labeled Faces in the Wild Home](http://vis-www.cs.umass.edu/lfw/lfw.tgz)
  
  Firstly, you should put the folder "stmodels" in "newpatch_rl\rlpatch".
  
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
    
  
```bash
$ cd rlpatch
$ python target_attack.py
```
