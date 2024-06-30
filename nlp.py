from flask import Flask ,request ,jsonify #line:1
from sentence_transformers import SentenceTransformer #line:2
from sklearn .cluster import KMeans #line:3
import numpy as np #line:4
app =Flask (__name__ )#line:6
model =SentenceTransformer ('xlm-r-bert-base-nli-stsb-mean-tokens')#line:8
def cluster_sentences (O00O0OOO0OOOO00O0 ,num_clusters =2 ):#line:10
    O00OO000000000OOO =model .encode (O00O0OOO0OOOO00O0 ,show_progress_bar =False )#line:11
    OOOOOOOOO0000O000 =KMeans (n_clusters =num_clusters )#line:12
    OOOOOOOOO0000O000 .fit (O00OO000000000OOO )#line:13
    O000O0OO0O00O0O0O =OOOOOOOOO0000O000 .cluster_centers_ #line:14
    O0OO0OOOOO0000OO0 =OOOOOOOOO0000O000 .labels_ #line:15
    return O000O0OO0O00O0O0O ,O0OO0OOOOO0000OO0 #line:16
def summarize_sentences (OO00O0OOOOO000000 ,num_clusters =2 ,num_sentences_per_cluster =1 ):#line:18
    O0O00O0OOO0O0O0O0 ,OO00O000O0O0O0O0O =cluster_sentences (OO00O0OOOOO000000 ,num_clusters )#line:19
    O00O0O00OOOO00000 =[]#line:20
    for O00000O0000OOOOOO in range (num_clusters ):#line:22
        OO00O0OOO00O0OOO0 =[OO00O0OOOOO000000 [O00OO000O000000OO ]for O00OO000O000000OO ,OOO0O0O0OO000O00O in enumerate (OO00O000O0O0O0O0O )if OOO0O0O0OO000O00O ==O00000O0000OOOOOO ]#line:23
        if OO00O0OOO00O0OOO0 :#line:24
            O00O0O00OOOO00000 .extend (OO00O0OOO00O0OOO0 [:num_sentences_per_cluster ])#line:25
    return O00O0O00OOOO00000 #line:27
@app .route ('/summarize',methods =['POST'])#line:29
def summarize_text ():#line:30
    O0O0OOO0OOO000O0O =request .get_json ()#line:31
    OOOO0OOO0OO0OOO00 =O0O0OOO0OOO000O0O ['text']#line:33
    try :#line:35
        O0O0OO0O00OO0OO0O =OOOO0OOO0OO0OOO00 .split ('.')#line:36
        OO0O00O000O000O00 =summarize_sentences (O0O0OO0O00OO0OO0O ,num_clusters =2 ,num_sentences_per_cluster =1 )#line:37
        OOO0OO00OOO0O0O0O =' '.join (OO0O00O000O000O00 )#line:39
        return jsonify ({'summary':OOO0OO00OOO0O0O0O })#line:41
    except Exception as O0O0OOO00OOO0O0O0 :#line:43
        return jsonify ({'error':str (O0O0OOO00OOO0O0O0 )}),500 #line:44
if __name__ =='__main__':#line:46
    app .run (debug =True )
