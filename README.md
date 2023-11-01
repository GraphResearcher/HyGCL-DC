# HyGCL-DC

Hypergraph Contrastive Learning for Drug Trafficking Community Detection
====
Official source code of "Hypergraph Contrastive Learning for Drug Trafficking Community Detection" 
(ICDM 2023) [[Source code](https://github.com/GraphResearcher/HyGCL-DC)]

[![Conference](https://img.shields.io/badge/ICDM-2023-blue)](https://www.cloud-conf.net/icdm2023/index.html)

<div>
<img src="https://github.com/GraphResearcher/HyGCL-DC/blob/main/figs/framework.png" width="1200" height="400">
<p>Fig. 1: The overall framework HyGCL-DC: DC: (a) it first constructs a hypergraph G based on the interactions among online
drug-related users; (b) it integrates augmentations from the structure view and the attribute view to augment hypergraphs into
 ̃G1 and  ̃G2. HyGCL-DC is designed as an end-to-end framework that integrates self-supervised contrastive learning to boost the
node embeddings over unlabeled data by reaching the agreement among positive and negative embedding pairs and supervised
learning with community labels for downstream drug trafficking community detetection.</p>
</div>


## Requirements

This code is developed and tested with python 3.11.1 and the required packages are listed in the `requirements.txt`.

Please run `pip install -r requirements.txt` to install all the dependencies. 


## Twitter-HyDrug
### New Dataset

Twitter-HyDrug is a real-world hypergraph data that describes the drug trafficking communities on Twitter. 
We first crawl the metadata (275,884,694 posts and 40,780,721 users) through the official
Twitter API from Dec 2020 to Aug 2021. 
Afterward, we generate a drug keyword list that covers 21 drug types that may cause drug overdose or drug addiction 
problems to filter the tweets that contain drug-relevant information. Based on the keyword list, we obtain 266,975
filtered drug-relevant posts by 54,680 users.
Moreover, we define six types of drug communities, i.e., cannabis, opioid, hallucinogen, stimulant, depressant, and
others communities, based on the drug functions.
Six researchers spent 62 days annotating these Twitter users into six communities based on the annotation rules discussed in the next section.
With the specific criteria, six researchers annotated the filtered metadata separately. For these Twitter users with disagreed labels, 
we conducted further discussion among annotators for cross-validation. To conclude, we obtained the drug hypergraph including 2,936 user nodes and 33,892 hyperedges.


### Annotation Rules
(i) If a user actively promotes some type of drug on Twitter or has rich connections (e.g.,
following, replying, liking, and retweeting) with other drug-related users in specific drug communities, he/she will be 
considered a member of the corresponding drug communities. For instance, Fig. 2(b) shows a drug seller who advertises his/her
drugs including oxycodone, cocaine, and Xanax. Based on the function of these drugs, we consider this user a member of
the overlapping communities including the opioid community (oxycodone), stimulant community (cocaine), and depressant
community (Xanax). (ii) If a user appears to suffer from drug overdose or drug addiction to the specific drug, he/she will
be regarded as a member of the specific drug community. For instance, Fig. 2(c) illustrates a drug user ”Ua\*\*\*on” that
suffers from an opioid overdose. So we classify this user as a member of the opioid community. (iii) If we can find evidence
on Twitter that a user used to purchase specific drugs from others on Twitter, then he/she belongs to a member of the
corresponding drug community. For example, Fig. 2(c) shows a drug buyer ”Bd\*\*\*in” that purchased oxycodone from drug
seller ”Su\*\*\*oy”. We consider he/she as a member of the opioid community. (iv) If we could not find any evidence that
a user suffers from drug overdoses or purchases from others, instead, he/she is very actively involved in discussing and
propagating specific drugs, we also consider him/her a member of the corresponding drug community. For instance, if a user
actively retweets LSD promotion tweets, but does not show evidence of purchasing or having LSD on Twitter, he/she is
still regarded as a member of the hallucinogen community. (v) If a user is promoting, purchasing or discussing
drugs on Twitter, but he/she does not mention the specific type of drugs, then we consider the user a member of another drug
community. For instance, if a user complains that he/she is suffering from drug overdoses but does not mention the type
of drugs that he/she is addicted to, then we regard the user as a member of another community. Mention that, partial of drugs
and the corresponding communities are listed in Table I. Based on the above strategy, we can obtain the ground truth
for the drug trafficking community detection task.


### Content Feature
For each Twitter user, we consider as a node in Twitter-HyDrug. Then, for each Twitter user, we extract the text information, including
usernames, profiles, and tweets to describe these users within drug communities. To further accurately characterize Twitter
users, we categorize the tweets into two groups: drug-related and drug-unrelated tweets, with a set of drug keywords. 
Then, as most users have sufficient tweets on Twitter, we select partial tweets including all drug-related
tweets and five drug-unrelated tweets to effectively and efficiently represent these users. Specifically, for drug-unrelated
tweets, we prioritize drug-unrelated tweets involving active interactions with other users in the drug community. For
instance, if the number of drug-unrelated tweets is much more than five, then we select these tweets with more interactions.
On the contrary, if the user has fewer than five drug-unrelated tweets, we leverage all the drug-unrelated tweets. Then
we combine profiles, usernames, drug-related tweets, as well as drug-unrelated tweets, and further feed them to the pre-trained transformer-based language model, SentenceBert, to obtain fixed-length feature vectors.

### Hyperedge
To exhaustively depict the complex and group-wise relationships among users in Twitter-HyDrug, we define
four types of hyperedges for describing the activities among users as follows: (i) R1: users-follow-user hyperedge 
relation denotes that a group of users follow a specific user in Twitter-HyDrug. The follow/following-based hyperedge 
aims to represent the social connections within drug trafficking communities, illustrating the friend circles involved 
in such illicit activities. (ii) R2: users-engage-conversation hyperedge relation represents that a group of users is 
engaged in a tweet-based conversation, encompassing activities such as posting, replying, retweeting, and liking the 
tweets involved within the conversation. The conversation-based hyperedge serves to portray the shared interests and 
topics among the group of users. (iii) R3: users-include-hashtag hyperedge relation
indicates that a bunch of users actively discuss the specific
hashtag-based topics by posting the specific hashtag in tweets
or profiles. Partial hashtag keywords are listed in Table I. For
instance, a group of users posts tweets on Twitter that include
oxycodone, one of the opioid drugs. Then the oxycodone hashtag will be considered as a hyperedge that encompasses all of the users in this group. (iv) R4: users-contain-emoji hyperedge relation signifies that a bunch of users contains a specific drug-related emoji in their tweets or profiles. 
Examples of drug-related emojis are illustrated in Fig. 2(b). Similar to hashtags,
we use emojis to describe the interested drugs in this group.



 
<div align="center">

| Community Type | Drugs (Key Words)                                   |
|----------------|-----------------------------------------------------|
 | Cannabis       | Cannabis, and all cannabis-infused products         |
| Opioid         | Oxycodone, Codeine, Morphine, Fentanyl, Hydrocodone |
| Hallucinogen   | LSD, MDMA, Shroom, Mescaline, DMT, Ketamine         |
| Stimulant      | Cocaine, Amphetamine, Metaphetamine                 | 
| Depressant     | Xanax, Farmapram, Valium, Halcion, Ativan, Klonopin |
| Others         | Drugs that are not listed above                     |
<p> Table I: Type of drug communities and related drugs </p>
</div>



<div align="center">
<img src="https://github.com/GraphResearcher/HyGCL-DC/blob/main/figs/Intro.jpg" width="500" height="500">
<p>Fig. 2: Illustration about drug trafficking communities among users on Twitter</p>
</div>

## Contact

Tianyi Ma - tma2@nd.edu 

Yiyue Qian - yqian5@nd.edu

Discussions, suggestions and questions are always welcome!

## Citation

```
@inproceedings{qianco,
  title={Hypergraph Contrastive Learning for Drug Trafficking Community Detection},
  author={Ma, Tianyi and Qian, Yiyue and Zhang, Chuxu  and Ye, Yanfang },
  booktitle={The IEEE International Conference on Data Mining},
  year={2023}
}
```


### Logger

This is a sample running logger which records the output and the model performance for Twitter-HyDrug data:

```
Epoch: 01, CL Loss: 8.6048, Train Loss: 1.0557, Valid Loss: 1.0148, Test  Loss: 0.9856, Train F1: 25.51%, Train Jaccard: 14.62%, Valid F1: 24.90%, Valid Jaccard: 14.22%, Test F1: 21.44%, Test Jaccard: 12.00%, 
Epoch: 02, CL Loss: 8.5773, Train Loss: 0.9634, Valid Loss: 0.9514, Test  Loss: 0.9001, Train F1: 31.18%, Train Jaccard: 18.47%, Valid F1: 30.05%, Valid Jaccard: 17.68%, Test F1: 25.95%, Test Jaccard: 14.91%, 
Epoch: 03, CL Loss: 8.5448, Train Loss: 0.9474, Valid Loss: 0.8659, Test  Loss: 0.9294, Train F1: 43.26%, Train Jaccard: 27.60%, Valid F1: 43.55%, Valid Jaccard: 27.83%, Test F1: 39.98%, Test Jaccard: 24.99%, 
Epoch: 04, CL Loss: 8.4773, Train Loss: 0.9074, Valid Loss: 0.8115, Test  Loss: 0.9186, Train F1: 55.23%, Train Jaccard: 38.15%, Valid F1: 56.16%, Valid Jaccard: 39.04%, Test F1: 56.68%, Test Jaccard: 39.54%, 
Epoch: 05, CL Loss: 8.4515, Train Loss: 0.8842, Valid Loss: 0.8282, Test  Loss: 0.9390, Train F1: 55.74%, Train Jaccard: 38.63%, Valid F1: 56.92%, Valid Jaccard: 39.78%, Test F1: 57.41%, Test Jaccard: 40.26%, 
Epoch: 06, CL Loss: 8.4089, Train Loss: 0.8451, Valid Loss: 0.7921, Test  Loss: 0.8905, Train F1: 55.64%, Train Jaccard: 38.55%, Valid F1: 55.74%, Valid Jaccard: 38.64%, Test F1: 59.08%, Test Jaccard: 41.92%, 
Epoch: 07, CL Loss: 8.3851, Train Loss: 0.8113, Valid Loss: 0.7904, Test  Loss: 0.8916, Train F1: 54.68%, Train Jaccard: 37.63%, Valid F1: 55.56%, Valid Jaccard: 38.47%, Test F1: 60.99%, Test Jaccard: 43.88%, 
Epoch: 08, CL Loss: 8.3366, Train Loss: 0.7748, Valid Loss: 0.8132, Test  Loss: 0.8674, Train F1: 55.11%, Train Jaccard: 38.03%, Valid F1: 55.45%, Valid Jaccard: 38.36%, Test F1: 61.77%, Test Jaccard: 44.69%, 
Epoch: 09, CL Loss: 8.3059, Train Loss: 0.7546, Valid Loss: 0.8093, Test  Loss: 0.8648, Train F1: 55.24%, Train Jaccard: 38.16%, Valid F1: 55.54%, Valid Jaccard: 38.44%, Test F1: 61.93%, Test Jaccard: 44.85%, 
Epoch: 10, CL Loss: 8.2579, Train Loss: 0.7431, Valid Loss: 0.7709, Test  Loss: 0.7838, Train F1: 55.34%, Train Jaccard: 38.25%, Valid F1: 55.75%, Valid Jaccard: 38.65%, Test F1: 61.92%, Test Jaccard: 44.84%, 
Epoch: 11, CL Loss: 8.2321, Train Loss: 0.7459, Valid Loss: 0.7659, Test  Loss: 0.7767, Train F1: 55.44%, Train Jaccard: 38.35%, Valid F1: 55.58%, Valid Jaccard: 38.48%, Test F1: 62.02%, Test Jaccard: 44.95%, 
Epoch: 12, CL Loss: 8.1817, Train Loss: 0.7355, Valid Loss: 0.7550, Test  Loss: 0.7623, Train F1: 55.59%, Train Jaccard: 38.50%, Valid F1: 55.56%, Valid Jaccard: 38.47%, Test F1: 61.88%, Test Jaccard: 44.81%, 
Epoch: 13, CL Loss: 8.1496, Train Loss: 0.7247, Valid Loss: 0.7488, Test  Loss: 0.7552, Train F1: 55.67%, Train Jaccard: 38.58%, Valid F1: 55.76%, Valid Jaccard: 38.66%, Test F1: 61.96%, Test Jaccard: 44.88%, 
Epoch: 14, CL Loss: 8.1081, Train Loss: 0.7123, Valid Loss: 0.7346, Test  Loss: 0.7406, Train F1: 55.91%, Train Jaccard: 38.80%, Valid F1: 55.96%, Valid Jaccard: 38.85%, Test F1: 62.21%, Test Jaccard: 45.15%, 
Epoch: 15, CL Loss: 8.0853, Train Loss: 0.7142, Valid Loss: 0.7268, Test  Loss: 0.7330, Train F1: 56.34%, Train Jaccard: 39.22%, Valid F1: 56.69%, Valid Jaccard: 39.56%, Test F1: 62.38%, Test Jaccard: 45.32%, 
Epoch: 16, CL Loss: 8.0485, Train Loss: 0.6994, Valid Loss: 0.7106, Test  Loss: 0.7188, Train F1: 56.97%, Train Jaccard: 39.83%, Valid F1: 57.43%, Valid Jaccard: 40.28%, Test F1: 62.83%, Test Jaccard: 45.80%, 
Epoch: 17, CL Loss: 8.0321, Train Loss: 0.6869, Valid Loss: 0.7025, Test  Loss: 0.7117, Train F1: 57.22%, Train Jaccard: 40.07%, Valid F1: 57.63%, Valid Jaccard: 40.48%, Test F1: 62.85%, Test Jaccard: 45.82%, 
Epoch: 18, CL Loss: 8.0144, Train Loss: 0.6588, Valid Loss: 0.6852, Test  Loss: 0.6985, Train F1: 58.11%, Train Jaccard: 40.95%, Valid F1: 58.47%, Valid Jaccard: 41.32%, Test F1: 62.51%, Test Jaccard: 45.46%, 
Epoch: 19, CL Loss: 8.0018, Train Loss: 0.6533, Valid Loss: 0.6766, Test  Loss: 0.7164, Train F1: 58.52%, Train Jaccard: 41.37%, Valid F1: 59.06%, Valid Jaccard: 41.90%, Test F1: 61.89%, Test Jaccard: 44.81%, 
Epoch: 20, CL Loss: 7.9848, Train Loss: 0.6344, Valid Loss: 0.6636, Test  Loss: 0.6991, Train F1: 59.61%, Train Jaccard: 42.46%, Valid F1: 58.83%, Valid Jaccard: 41.67%, Test F1: 61.56%, Test Jaccard: 44.47%, 
Epoch: 21, CL Loss: 7.9787, Train Loss: 0.6307, Valid Loss: 0.6841, Test  Loss: 0.6739, Train F1: 59.81%, Train Jaccard: 42.66%, Valid F1: 59.03%, Valid Jaccard: 41.87%, Test F1: 60.90%, Test Jaccard: 43.78%, 
Epoch: 22, CL Loss: 7.9700, Train Loss: 0.6438, Valid Loss: 0.6784, Test  Loss: 0.7240, Train F1: 60.03%, Train Jaccard: 42.89%, Valid F1: 59.53%, Valid Jaccard: 42.38%, Test F1: 60.37%, Test Jaccard: 43.24%, 
Epoch: 23, CL Loss: 7.9676, Train Loss: 0.6419, Valid Loss: 0.6764, Test  Loss: 0.7185, Train F1: 60.25%, Train Jaccard: 43.11%, Valid F1: 60.02%, Valid Jaccard: 42.88%, Test F1: 60.85%, Test Jaccard: 43.73%, 
Epoch: 24, CL Loss: 7.9593, Train Loss: 0.6392, Valid Loss: 0.7016, Test  Loss: 0.7165, Train F1: 60.72%, Train Jaccard: 43.59%, Valid F1: 60.45%, Valid Jaccard: 43.32%, Test F1: 61.05%, Test Jaccard: 43.94%, 
Epoch: 25, CL Loss: 7.9522, Train Loss: 0.6290, Valid Loss: 0.7003, Test  Loss: 0.7185, Train F1: 60.65%, Train Jaccard: 43.52%, Valid F1: 60.30%, Valid Jaccard: 43.16%, Test F1: 60.78%, Test Jaccard: 43.65%, 
Epoch: 26, CL Loss: 7.9478, Train Loss: 0.6279, Valid Loss: 0.6983, Test  Loss: 0.7233, Train F1: 60.29%, Train Jaccard: 43.16%, Valid F1: 59.74%, Valid Jaccard: 42.60%, Test F1: 60.60%, Test Jaccard: 43.47%, 
Epoch: 27, CL Loss: 7.9444, Train Loss: 0.6274, Valid Loss: 0.6975, Test  Loss: 0.7267, Train F1: 60.07%, Train Jaccard: 42.93%, Valid F1: 59.67%, Valid Jaccard: 42.52%, Test F1: 60.53%, Test Jaccard: 43.40%, 
Epoch: 28, CL Loss: 7.9373, Train Loss: 0.6219, Valid Loss: 0.6957, Test  Loss: 0.7330, Train F1: 60.08%, Train Jaccard: 42.94%, Valid F1: 59.43%, Valid Jaccard: 42.28%, Test F1: 60.96%, Test Jaccard: 43.84%, 
Epoch: 29, CL Loss: 7.9377, Train Loss: 0.6105, Valid Loss: 0.6704, Test  Loss: 0.7360, Train F1: 60.10%, Train Jaccard: 42.96%, Valid F1: 59.48%, Valid Jaccard: 42.32%, Test F1: 61.11%, Test Jaccard: 43.99%, 
Epoch: 30, CL Loss: 7.9266, Train Loss: 0.6052, Valid Loss: 0.6665, Test  Loss: 0.7672, Train F1: 60.20%, Train Jaccard: 43.06%, Valid F1: 59.52%, Valid Jaccard: 42.36%, Test F1: 61.80%, Test Jaccard: 44.72%, 
Epoch: 31, CL Loss: 7.9273, Train Loss: 0.5960, Valid Loss: 0.6953, Test  Loss: 0.7509, Train F1: 60.13%, Train Jaccard: 42.99%, Valid F1: 59.55%, Valid Jaccard: 42.40%, Test F1: 62.09%, Test Jaccard: 45.02%, 
Epoch: 32, CL Loss: 7.9295, Train Loss: 0.5934, Valid Loss: 0.6996, Test  Loss: 0.8247, Train F1: 60.24%, Train Jaccard: 43.10%, Valid F1: 59.79%, Valid Jaccard: 42.64%, Test F1: 61.98%, Test Jaccard: 44.91%, 
Epoch: 33, CL Loss: 7.9266, Train Loss: 0.5928, Valid Loss: 0.7258, Test  Loss: 0.8474, Train F1: 60.25%, Train Jaccard: 43.12%, Valid F1: 59.94%, Valid Jaccard: 42.80%, Test F1: 62.16%, Test Jaccard: 45.10%, 
Epoch: 34, CL Loss: 7.9183, Train Loss: 0.5914, Valid Loss: 0.7273, Test  Loss: 0.8734, Train F1: 60.34%, Train Jaccard: 43.21%, Valid F1: 59.82%, Valid Jaccard: 42.67%, Test F1: 62.28%, Test Jaccard: 45.22%, 
Epoch: 35, CL Loss: 7.9195, Train Loss: 0.5908, Valid Loss: 0.7299, Test  Loss: 0.8940, Train F1: 60.25%, Train Jaccard: 43.11%, Valid F1: 59.75%, Valid Jaccard: 42.60%, Test F1: 62.20%, Test Jaccard: 45.14%, 
Epoch: 36, CL Loss: 7.9150, Train Loss: 0.5866, Valid Loss: 0.7568, Test  Loss: 0.8938, Train F1: 60.35%, Train Jaccard: 43.22%, Valid F1: 59.85%, Valid Jaccard: 42.70%, Test F1: 62.17%, Test Jaccard: 45.11%, 
Epoch: 37, CL Loss: 7.9147, Train Loss: 0.5843, Valid Loss: 0.7571, Test  Loss: 0.8937, Train F1: 60.39%, Train Jaccard: 43.26%, Valid F1: 59.89%, Valid Jaccard: 42.74%, Test F1: 62.10%, Test Jaccard: 45.03%, 
Epoch: 38, CL Loss: 7.9120, Train Loss: 0.5718, Valid Loss: 0.7290, Test  Loss: 0.8935, Train F1: 60.59%, Train Jaccard: 43.46%, Valid F1: 60.23%, Valid Jaccard: 43.09%, Test F1: 62.09%, Test Jaccard: 45.02%, 
Epoch: 39, CL Loss: 7.9131, Train Loss: 0.5670, Valid Loss: 0.7346, Test  Loss: 0.8932, Train F1: 60.68%, Train Jaccard: 43.55%, Valid F1: 60.22%, Valid Jaccard: 43.08%, Test F1: 62.10%, Test Jaccard: 45.03%, 
Epoch: 40, CL Loss: 7.9154, Train Loss: 0.5787, Valid Loss: 0.7572, Test  Loss: 0.9164, Train F1: 60.75%, Train Jaccard: 43.62%, Valid F1: 60.44%, Valid Jaccard: 43.31%, Test F1: 61.91%, Test Jaccard: 44.83%, 
Epoch: 41, CL Loss: 7.9093, Train Loss: 0.5785, Valid Loss: 0.7568, Test  Loss: 0.9158, Train F1: 60.76%, Train Jaccard: 43.64%, Valid F1: 60.28%, Valid Jaccard: 43.14%, Test F1: 61.65%, Test Jaccard: 44.56%, 
Epoch: 42, CL Loss: 7.9079, Train Loss: 0.5687, Valid Loss: 0.7548, Test  Loss: 0.9148, Train F1: 60.78%, Train Jaccard: 43.66%, Valid F1: 60.29%, Valid Jaccard: 43.16%, Test F1: 61.33%, Test Jaccard: 44.23%, 
Epoch: 43, CL Loss: 7.9084, Train Loss: 0.5688, Valid Loss: 0.7532, Test  Loss: 0.8922, Train F1: 60.77%, Train Jaccard: 43.64%, Valid F1: 60.21%, Valid Jaccard: 43.08%, Test F1: 61.05%, Test Jaccard: 43.93%, 
Epoch: 44, CL Loss: 7.9010, Train Loss: 0.5697, Valid Loss: 0.7504, Test  Loss: 0.8898, Train F1: 60.81%, Train Jaccard: 43.69%, Valid F1: 60.34%, Valid Jaccard: 43.21%, Test F1: 61.10%, Test Jaccard: 43.99%, 
Epoch: 45, CL Loss: 7.9026, Train Loss: 0.5700, Valid Loss: 0.7492, Test  Loss: 0.8894, Train F1: 60.88%, Train Jaccard: 43.76%, Valid F1: 60.49%, Valid Jaccard: 43.36%, Test F1: 61.18%, Test Jaccard: 44.07%, 
Epoch: 46, CL Loss: 7.9035, Train Loss: 0.5705, Valid Loss: 0.7228, Test  Loss: 0.8885, Train F1: 60.90%, Train Jaccard: 43.78%, Valid F1: 60.57%, Valid Jaccard: 43.44%, Test F1: 61.62%, Test Jaccard: 44.53%, 
Epoch: 47, CL Loss: 7.9023, Train Loss: 0.5698, Valid Loss: 0.6918, Test  Loss: 0.8673, Train F1: 60.88%, Train Jaccard: 43.77%, Valid F1: 60.56%, Valid Jaccard: 43.43%, Test F1: 61.65%, Test Jaccard: 44.56%, 
Epoch: 48, CL Loss: 7.8971, Train Loss: 0.5685, Valid Loss: 0.6628, Test  Loss: 0.8179, Train F1: 60.83%, Train Jaccard: 43.71%, Valid F1: 60.45%, Valid Jaccard: 43.32%, Test F1: 61.79%, Test Jaccard: 44.70%, 
Epoch: 49, CL Loss: 7.8967, Train Loss: 0.5679, Valid Loss: 0.6648, Test  Loss: 0.8204, Train F1: 60.89%, Train Jaccard: 43.78%, Valid F1: 60.41%, Valid Jaccard: 43.28%, Test F1: 61.84%, Test Jaccard: 44.76%, 
Epoch: 50, CL Loss: 7.8946, Train Loss: 0.5574, Valid Loss: 0.6677, Test  Loss: 0.8423, Train F1: 60.93%, Train Jaccard: 43.81%, Valid F1: 60.45%, Valid Jaccard: 43.31%, Test F1: 61.88%, Test Jaccard: 44.80%, 
Epoch: 51, CL Loss: 7.8946, Train Loss: 0.5568, Valid Loss: 0.6683, Test  Loss: 0.8424, Train F1: 60.95%, Train Jaccard: 43.83%, Valid F1: 60.52%, Valid Jaccard: 43.39%, Test F1: 61.93%, Test Jaccard: 44.86%, 
Epoch: 52, CL Loss: 7.8930, Train Loss: 0.5558, Valid Loss: 0.6924, Test  Loss: 0.8195, Train F1: 61.06%, Train Jaccard: 43.94%, Valid F1: 60.55%, Valid Jaccard: 43.42%, Test F1: 62.28%, Test Jaccard: 45.22%, 
Epoch: 53, CL Loss: 7.8943, Train Loss: 0.5646, Valid Loss: 0.6928, Test  Loss: 0.8190, Train F1: 61.08%, Train Jaccard: 43.97%, Valid F1: 60.54%, Valid Jaccard: 43.41%, Test F1: 62.40%, Test Jaccard: 45.35%, 
Epoch: 54, CL Loss: 7.8929, Train Loss: 0.5636, Valid Loss: 0.6928, Test  Loss: 0.8194, Train F1: 61.18%, Train Jaccard: 44.08%, Valid F1: 60.54%, Valid Jaccard: 43.41%, Test F1: 62.50%, Test Jaccard: 45.45%, 
Epoch: 55, CL Loss: 7.8934, Train Loss: 0.5631, Valid Loss: 0.6921, Test  Loss: 0.8182, Train F1: 61.25%, Train Jaccard: 44.14%, Valid F1: 60.59%, Valid Jaccard: 43.46%, Test F1: 62.54%, Test Jaccard: 45.49%, 
Epoch: 56, CL Loss: 7.8925, Train Loss: 0.5622, Valid Loss: 0.6889, Test  Loss: 0.7979, Train F1: 61.28%, Train Jaccard: 44.17%, Valid F1: 60.71%, Valid Jaccard: 43.58%, Test F1: 62.37%, Test Jaccard: 45.32%, 
Epoch: 57, CL Loss: 7.8912, Train Loss: 0.5616, Valid Loss: 0.6886, Test  Loss: 0.7974, Train F1: 61.34%, Train Jaccard: 44.24%, Valid F1: 60.74%, Valid Jaccard: 43.62%, Test F1: 62.35%, Test Jaccard: 45.29%, 
Epoch: 58, CL Loss: 7.8908, Train Loss: 0.5606, Valid Loss: 0.7165, Test  Loss: 0.8166, Train F1: 61.47%, Train Jaccard: 44.38%, Valid F1: 60.73%, Valid Jaccard: 43.61%, Test F1: 62.58%, Test Jaccard: 45.54%, 
Epoch: 59, CL Loss: 7.8868, Train Loss: 0.5601, Valid Loss: 0.6879, Test  Loss: 0.8183, Train F1: 61.49%, Train Jaccard: 44.39%, Valid F1: 60.87%, Valid Jaccard: 43.75%, Test F1: 62.62%, Test Jaccard: 45.58%, 
Epoch: 60, CL Loss: 7.8866, Train Loss: 0.5590, Valid Loss: 0.6880, Test  Loss: 0.8634, Train F1: 61.61%, Train Jaccard: 44.52%, Valid F1: 60.86%, Valid Jaccard: 43.74%, Test F1: 62.83%, Test Jaccard: 45.80%, 
Epoch: 61, CL Loss: 7.8866, Train Loss: 0.5585, Valid Loss: 0.6881, Test  Loss: 0.8621, Train F1: 61.74%, Train Jaccard: 44.66%, Valid F1: 60.83%, Valid Jaccard: 43.71%, Test F1: 62.75%, Test Jaccard: 45.72%, 
Epoch: 62, CL Loss: 7.8830, Train Loss: 0.5576, Valid Loss: 0.6874, Test  Loss: 0.8595, Train F1: 61.98%, Train Jaccard: 44.91%, Valid F1: 61.22%, Valid Jaccard: 44.12%, Test F1: 63.08%, Test Jaccard: 46.07%, 
Epoch: 63, CL Loss: 7.8830, Train Loss: 0.5571, Valid Loss: 0.6869, Test  Loss: 0.8583, Train F1: 62.07%, Train Jaccard: 45.00%, Valid F1: 61.29%, Valid Jaccard: 44.19%, Test F1: 63.08%, Test Jaccard: 46.08%, 
Epoch: 64, CL Loss: 7.8820, Train Loss: 0.5560, Valid Loss: 0.6845, Test  Loss: 0.8564, Train F1: 62.16%, Train Jaccard: 45.09%, Valid F1: 61.46%, Valid Jaccard: 44.37%, Test F1: 63.28%, Test Jaccard: 46.28%, 
Epoch: 65, CL Loss: 7.8806, Train Loss: 0.5553, Valid Loss: 0.6830, Test  Loss: 0.8556, Train F1: 62.23%, Train Jaccard: 45.16%, Valid F1: 61.54%, Valid Jaccard: 44.44%, Test F1: 63.47%, Test Jaccard: 46.49%, 
Epoch: 66, CL Loss: 7.8852, Train Loss: 0.5544, Valid Loss: 0.6856, Test  Loss: 0.8775, Train F1: 62.20%, Train Jaccard: 45.13%, Valid F1: 61.36%, Valid Jaccard: 44.26%, Test F1: 63.73%, Test Jaccard: 46.77%, 
Epoch: 67, CL Loss: 7.8823, Train Loss: 0.5539, Valid Loss: 0.6870, Test  Loss: 0.8324, Train F1: 62.16%, Train Jaccard: 45.09%, Valid F1: 61.18%, Valid Jaccard: 44.07%, Test F1: 63.77%, Test Jaccard: 46.81%, 
Epoch: 68, CL Loss: 7.8815, Train Loss: 0.5434, Valid Loss: 0.6891, Test  Loss: 0.8310, Train F1: 62.28%, Train Jaccard: 45.22%, Valid F1: 61.21%, Valid Jaccard: 44.10%, Test F1: 63.70%, Test Jaccard: 46.74%, 
Epoch: 69, CL Loss: 7.8804, Train Loss: 0.5429, Valid Loss: 0.6884, Test  Loss: 0.8303, Train F1: 62.21%, Train Jaccard: 45.15%, Valid F1: 60.87%, Valid Jaccard: 43.75%, Test F1: 63.97%, Test Jaccard: 47.03%, 
Epoch: 70, CL Loss: 7.8806, Train Loss: 0.5418, Valid Loss: 0.6872, Test  Loss: 0.8288, Train F1: 61.78%, Train Jaccard: 44.69%, Valid F1: 61.09%, Valid Jaccard: 43.98%, Test F1: 64.40%, Test Jaccard: 47.49%, 
Epoch: 71, CL Loss: 7.8829, Train Loss: 0.5413, Valid Loss: 0.6866, Test  Loss: 0.8281, Train F1: 61.64%, Train Jaccard: 44.55%, Valid F1: 60.76%, Valid Jaccard: 43.64%, Test F1: 64.35%, Test Jaccard: 47.44%, 
Epoch: 72, CL Loss: 7.8779, Train Loss: 0.5400, Valid Loss: 0.6313, Test  Loss: 0.8271, Train F1: 61.55%, Train Jaccard: 44.45%, Valid F1: 60.60%, Valid Jaccard: 43.48%, Test F1: 64.38%, Test Jaccard: 47.47%, 
Epoch: 73, CL Loss: 7.8791, Train Loss: 0.5394, Valid Loss: 0.6296, Test  Loss: 0.8267, Train F1: 61.53%, Train Jaccard: 44.44%, Valid F1: 60.63%, Valid Jaccard: 43.51%, Test F1: 64.54%, Test Jaccard: 47.64%, 
Epoch: 74, CL Loss: 7.8776, Train Loss: 0.5380, Valid Loss: 0.6269, Test  Loss: 0.8262, Train F1: 61.72%, Train Jaccard: 44.63%, Valid F1: 60.66%, Valid Jaccard: 43.54%, Test F1: 64.64%, Test Jaccard: 47.75%, 
Epoch: 75, CL Loss: 7.8765, Train Loss: 0.5466, Valid Loss: 0.6262, Test  Loss: 0.8260, Train F1: 61.88%, Train Jaccard: 44.80%, Valid F1: 60.81%, Valid Jaccard: 43.69%, Test F1: 64.74%, Test Jaccard: 47.86%, 
Epoch: 76, CL Loss: 7.8759, Train Loss: 0.5451, Valid Loss: 0.6250, Test  Loss: 0.8255, Train F1: 62.20%, Train Jaccard: 45.14%, Valid F1: 61.03%, Valid Jaccard: 43.91%, Test F1: 64.81%, Test Jaccard: 47.94%, 
Epoch: 77, CL Loss: 7.8800, Train Loss: 0.5443, Valid Loss: 0.6244, Test  Loss: 0.8253, Train F1: 62.30%, Train Jaccard: 45.24%, Valid F1: 61.42%, Valid Jaccard: 44.32%, Test F1: 65.02%, Test Jaccard: 48.17%, 
Epoch: 78, CL Loss: 7.8752, Train Loss: 0.5427, Valid Loss: 0.6232, Test  Loss: 0.8038, Train F1: 62.56%, Train Jaccard: 45.52%, Valid F1: 61.58%, Valid Jaccard: 44.49%, Test F1: 65.16%, Test Jaccard: 48.32%, 
Epoch: 79, CL Loss: 7.8749, Train Loss: 0.5419, Valid Loss: 0.6226, Test  Loss: 0.8246, Train F1: 62.67%, Train Jaccard: 45.63%, Valid F1: 62.00%, Valid Jaccard: 44.93%, Test F1: 65.07%, Test Jaccard: 48.22%, 
Epoch: 80, CL Loss: 7.8789, Train Loss: 0.5402, Valid Loss: 0.6229, Test  Loss: 0.8032, Train F1: 62.97%, Train Jaccard: 45.95%, Valid F1: 62.25%, Valid Jaccard: 45.19%, Test F1: 64.73%, Test Jaccard: 47.85%, 
Epoch: 81, CL Loss: 7.8749, Train Loss: 0.5394, Valid Loss: 0.6234, Test  Loss: 0.8026, Train F1: 63.12%, Train Jaccard: 46.11%, Valid F1: 62.33%, Valid Jaccard: 45.27%, Test F1: 64.54%, Test Jaccard: 47.64%, 
Epoch: 82, CL Loss: 7.8728, Train Loss: 0.5376, Valid Loss: 0.6239, Test  Loss: 0.8010, Train F1: 63.31%, Train Jaccard: 46.32%, Valid F1: 62.31%, Valid Jaccard: 45.26%, Test F1: 64.81%, Test Jaccard: 47.94%, 
Epoch: 83, CL Loss: 7.8750, Train Loss: 0.5368, Valid Loss: 0.6471, Test  Loss: 0.7999, Train F1: 63.43%, Train Jaccard: 46.45%, Valid F1: 62.42%, Valid Jaccard: 45.37%, Test F1: 64.94%, Test Jaccard: 48.08%, 
Epoch: 84, CL Loss: 7.8697, Train Loss: 0.5351, Valid Loss: 0.6457, Test  Loss: 0.7984, Train F1: 63.59%, Train Jaccard: 46.62%, Valid F1: 62.54%, Valid Jaccard: 45.49%, Test F1: 65.07%, Test Jaccard: 48.23%, 
Epoch: 85, CL Loss: 7.8756, Train Loss: 0.5342, Valid Loss: 0.6449, Test  Loss: 0.7980, Train F1: 63.64%, Train Jaccard: 46.67%, Valid F1: 62.67%, Valid Jaccard: 45.63%, Test F1: 65.17%, Test Jaccard: 48.33%, 
Epoch: 86, CL Loss: 7.8697, Train Loss: 0.5325, Valid Loss: 0.6434, Test  Loss: 0.7972, Train F1: 63.72%, Train Jaccard: 46.75%, Valid F1: 62.67%, Valid Jaccard: 45.63%, Test F1: 65.27%, Test Jaccard: 48.44%, 
Epoch: 87, CL Loss: 7.8728, Train Loss: 0.5316, Valid Loss: 0.6427, Test  Loss: 0.7968, Train F1: 63.72%, Train Jaccard: 46.76%, Valid F1: 62.67%, Valid Jaccard: 45.63%, Test F1: 65.56%, Test Jaccard: 48.76%, 
Epoch: 88, CL Loss: 7.8716, Train Loss: 0.5297, Valid Loss: 0.6411, Test  Loss: 0.7959, Train F1: 63.81%, Train Jaccard: 46.85%, Valid F1: 62.67%, Valid Jaccard: 45.63%, Test F1: 65.74%, Test Jaccard: 48.97%, 
Epoch: 89, CL Loss: 7.8681, Train Loss: 0.5288, Valid Loss: 0.6118, Test  Loss: 0.7955, Train F1: 63.79%, Train Jaccard: 46.84%, Valid F1: 62.80%, Valid Jaccard: 45.77%, Test F1: 65.81%, Test Jaccard: 49.04%, 
Epoch: 90, CL Loss: 7.8699, Train Loss: 0.5269, Valid Loss: 0.6386, Test  Loss: 0.7948, Train F1: 63.96%, Train Jaccard: 47.02%, Valid F1: 62.79%, Valid Jaccard: 45.76%, Test F1: 66.06%, Test Jaccard: 49.32%, 
Epoch: 91, CL Loss: 7.8726, Train Loss: 0.5260, Valid Loss: 0.6114, Test  Loss: 0.7944, Train F1: 64.07%, Train Jaccard: 47.13%, Valid F1: 62.91%, Valid Jaccard: 45.89%, Test F1: 66.16%, Test Jaccard: 49.43%, 
Epoch: 92, CL Loss: 7.8684, Train Loss: 0.5241, Valid Loss: 0.6035, Test  Loss: 0.7692, Train F1: 64.11%, Train Jaccard: 47.18%, Valid F1: 63.01%, Valid Jaccard: 45.99%, Test F1: 66.38%, Test Jaccard: 49.68%, 
Epoch: 93, CL Loss: 7.8709, Train Loss: 0.5231, Valid Loss: 0.6022, Test  Loss: 0.7686, Train F1: 64.12%, Train Jaccard: 47.19%, Valid F1: 63.04%, Valid Jaccard: 46.03%, Test F1: 66.44%, Test Jaccard: 49.75%, 
Epoch: 94, CL Loss: 7.8679, Train Loss: 0.5213, Valid Loss: 0.5997, Test  Loss: 0.7676, Train F1: 64.16%, Train Jaccard: 47.23%, Valid F1: 63.23%, Valid Jaccard: 46.23%, Test F1: 66.83%, Test Jaccard: 50.18%, 
Epoch: 95, CL Loss: 7.8727, Train Loss: 0.5203, Valid Loss: 0.5984, Test  Loss: 0.7671, Train F1: 64.20%, Train Jaccard: 47.28%, Valid F1: 63.23%, Valid Jaccard: 46.23%, Test F1: 66.95%, Test Jaccard: 50.32%, 
Epoch: 96, CL Loss: 7.8832, Train Loss: 0.5184, Valid Loss: 0.6247, Test  Loss: 0.7658, Train F1: 64.44%, Train Jaccard: 47.54%, Valid F1: 63.40%, Valid Jaccard: 46.41%, Test F1: 67.11%, Test Jaccard: 50.51%, 
Epoch: 97, CL Loss: 7.8766, Train Loss: 0.5175, Valid Loss: 0.6237, Test  Loss: 0.7649, Train F1: 64.52%, Train Jaccard: 47.62%, Valid F1: 63.53%, Valid Jaccard: 46.56%, Test F1: 66.95%, Test Jaccard: 50.32%, 
Epoch: 98, CL Loss: 7.8813, Train Loss: 0.5157, Valid Loss: 0.6219, Test  Loss: 0.7633, Train F1: 64.63%, Train Jaccard: 47.74%, Valid F1: 63.60%, Valid Jaccard: 46.63%, Test F1: 67.02%, Test Jaccard: 50.40%, 
Epoch: 99, CL Loss: 7.8952, Train Loss: 0.5148, Valid Loss: 0.6210, Test  Loss: 0.7627, Train F1: 64.65%, Train Jaccard: 47.77%, Valid F1: 63.74%, Valid Jaccard: 46.77%, Test F1: 67.08%, Test Jaccard: 50.47%, 
Epoch: 100, CL Loss: 7.8981, Train Loss: 0.5130, Valid Loss: 0.6191, Test  Loss: 0.7612, Train F1: 64.75%, Train Jaccard: 47.87%, Valid F1: 64.00%, Valid Jaccard: 47.06%, Test F1: 67.40%, Test Jaccard: 50.83%, 
Epoch: 101, CL Loss: 7.8727, Train Loss: 0.5120, Valid Loss: 0.6181, Test  Loss: 0.7604, Train F1: 64.78%, Train Jaccard: 47.91%, Valid F1: 64.00%, Valid Jaccard: 47.06%, Test F1: 67.43%, Test Jaccard: 50.86%, 
Epoch: 102, CL Loss: 7.8706, Train Loss: 0.5101, Valid Loss: 0.6161, Test  Loss: 0.7593, Train F1: 64.84%, Train Jaccard: 47.98%, Valid F1: 64.24%, Valid Jaccard: 47.31%, Test F1: 67.40%, Test Jaccard: 50.83%, 
Epoch: 103, CL Loss: 7.8792, Train Loss: 0.5091, Valid Loss: 0.6151, Test  Loss: 0.7596, Train F1: 64.97%, Train Jaccard: 48.12%, Valid F1: 64.21%, Valid Jaccard: 47.28%, Test F1: 67.46%, Test Jaccard: 50.90%, 
Epoch: 104, CL Loss: 7.8726, Train Loss: 0.5071, Valid Loss: 0.5892, Test  Loss: 0.7601, Train F1: 65.30%, Train Jaccard: 48.48%, Valid F1: 64.25%, Valid Jaccard: 47.33%, Test F1: 67.52%, Test Jaccard: 50.97%, 
Epoch: 105, CL Loss: 7.8685, Train Loss: 0.5061, Valid Loss: 0.5882, Test  Loss: 0.7602, Train F1: 65.44%, Train Jaccard: 48.64%, Valid F1: 64.38%, Valid Jaccard: 47.47%, Test F1: 67.59%, Test Jaccard: 51.04%, 
Epoch: 106, CL Loss: 7.8728, Train Loss: 0.5040, Valid Loss: 0.5862, Test  Loss: 0.7838, Train F1: 65.57%, Train Jaccard: 48.78%, Valid F1: 64.45%, Valid Jaccard: 47.54%, Test F1: 67.65%, Test Jaccard: 51.11%, 
Epoch: 107, CL Loss: 7.8705, Train Loss: 0.5029, Valid Loss: 0.5849, Test  Loss: 0.8045, Train F1: 65.71%, Train Jaccard: 48.93%, Valid F1: 64.68%, Valid Jaccard: 47.79%, Test F1: 67.80%, Test Jaccard: 51.29%, 
Epoch: 108, CL Loss: 7.8669, Train Loss: 0.5009, Valid Loss: 0.5825, Test  Loss: 0.8037, Train F1: 65.96%, Train Jaccard: 49.21%, Valid F1: 65.17%, Valid Jaccard: 48.34%, Test F1: 68.12%, Test Jaccard: 51.65%, 
Epoch: 109, CL Loss: 7.8711, Train Loss: 0.4998, Valid Loss: 0.5814, Test  Loss: 0.8033, Train F1: 66.09%, Train Jaccard: 49.35%, Valid F1: 65.47%, Valid Jaccard: 48.66%, Test F1: 68.18%, Test Jaccard: 51.72%, 
Epoch: 110, CL Loss: 7.8680, Train Loss: 0.4978, Valid Loss: 0.5790, Test  Loss: 0.8026, Train F1: 66.48%, Train Jaccard: 49.79%, Valid F1: 66.05%, Valid Jaccard: 49.31%, Test F1: 68.58%, Test Jaccard: 52.19%, 
Epoch: 111, CL Loss: 7.8664, Train Loss: 0.4968, Valid Loss: 0.5779, Test  Loss: 0.8023, Train F1: 66.75%, Train Jaccard: 50.09%, Valid F1: 66.44%, Valid Jaccard: 49.75%, Test F1: 68.52%, Test Jaccard: 52.11%, 
Epoch: 112, CL Loss: 7.8650, Train Loss: 0.4948, Valid Loss: 0.5758, Test  Loss: 0.8211, Train F1: 67.47%, Train Jaccard: 50.91%, Valid F1: 67.12%, Valid Jaccard: 50.51%, Test F1: 68.91%, Test Jaccard: 52.57%, 
Epoch: 113, CL Loss: 7.8635, Train Loss: 0.4937, Valid Loss: 0.5748, Test  Loss: 0.8199, Train F1: 67.68%, Train Jaccard: 51.15%, Valid F1: 67.18%, Valid Jaccard: 50.58%, Test F1: 69.13%, Test Jaccard: 52.82%, 
Epoch: 114, CL Loss: 7.8641, Train Loss: 0.4918, Valid Loss: 0.5732, Test  Loss: 0.8170, Train F1: 67.96%, Train Jaccard: 51.46%, Valid F1: 67.37%, Valid Jaccard: 50.79%, Test F1: 69.21%, Test Jaccard: 52.92%, 
Epoch: 115, CL Loss: 7.8624, Train Loss: 0.4909, Valid Loss: 0.5726, Test  Loss: 0.8153, Train F1: 68.27%, Train Jaccard: 51.83%, Valid F1: 67.75%, Valid Jaccard: 51.23%, Test F1: 69.46%, Test Jaccard: 53.21%, 
Epoch: 116, CL Loss: 7.8684, Train Loss: 0.4890, Valid Loss: 0.5710, Test  Loss: 0.8382, Train F1: 68.61%, Train Jaccard: 52.22%, Valid F1: 67.94%, Valid Jaccard: 51.44%, Test F1: 69.63%, Test Jaccard: 53.41%, 
Epoch: 117, CL Loss: 7.8640, Train Loss: 0.4880, Valid Loss: 0.5702, Test  Loss: 0.8379, Train F1: 68.69%, Train Jaccard: 52.32%, Valid F1: 68.12%, Valid Jaccard: 51.66%, Test F1: 69.65%, Test Jaccard: 53.44%, 
Epoch: 118, CL Loss: 7.8617, Train Loss: 0.4868, Valid Loss: 0.5685, Test  Loss: 0.8574, Train F1: 68.78%, Train Jaccard: 52.42%, Valid F1: 68.21%, Valid Jaccard: 51.76%, Test F1: 69.95%, Test Jaccard: 53.79%, 
Epoch: 119, CL Loss: 7.8630, Train Loss: 0.4943, Valid Loss: 0.5676, Test  Loss: 0.8357, Train F1: 68.86%, Train Jaccard: 52.51%, Valid F1: 68.28%, Valid Jaccard: 51.83%, Test F1: 70.05%, Test Jaccard: 53.90%, 
Epoch: 120, CL Loss: 7.8630, Train Loss: 0.4829, Valid Loss: 0.5375, Test  Loss: 0.8341, Train F1: 68.98%, Train Jaccard: 52.65%, Valid F1: 68.43%, Valid Jaccard: 52.01%, Test F1: 70.25%, Test Jaccard: 54.15%, 
Epoch: 121, CL Loss: 7.8590, Train Loss: 0.4818, Valid Loss: 0.5646, Test  Loss: 0.8341, Train F1: 68.98%, Train Jaccard: 52.64%, Valid F1: 68.68%, Valid Jaccard: 52.31%, Test F1: 70.29%, Test Jaccard: 54.18%, 
Epoch: 122, CL Loss: 7.8646, Train Loss: 0.4799, Valid Loss: 0.5619, Test  Loss: 0.8547, Train F1: 69.20%, Train Jaccard: 52.90%, Valid F1: 68.90%, Valid Jaccard: 52.56%, Test F1: 70.40%, Test Jaccard: 54.33%, 
Epoch: 123, CL Loss: 7.8634, Train Loss: 0.4789, Valid Loss: 0.5605, Test  Loss: 0.8546, Train F1: 69.27%, Train Jaccard: 52.99%, Valid F1: 69.06%, Valid Jaccard: 52.74%, Test F1: 70.56%, Test Jaccard: 54.51%, 
Epoch: 124, CL Loss: 7.8603, Train Loss: 0.4771, Valid Loss: 0.5574, Test  Loss: 0.8532, Train F1: 69.63%, Train Jaccard: 53.41%, Valid F1: 69.19%, Valid Jaccard: 52.89%, Test F1: 70.71%, Test Jaccard: 54.69%, 
Epoch: 125, CL Loss: 7.8629, Train Loss: 0.4763, Valid Loss: 0.5558, Test  Loss: 0.8527, Train F1: 69.68%, Train Jaccard: 53.46%, Valid F1: 69.38%, Valid Jaccard: 53.11%, Test F1: 70.96%, Test Jaccard: 54.99%, 
Epoch: 126, CL Loss: 7.8596, Train Loss: 0.4652, Valid Loss: 0.5532, Test  Loss: 0.8521, Train F1: 69.99%, Train Jaccard: 53.84%, Valid F1: 69.56%, Valid Jaccard: 53.33%, Test F1: 70.98%, Test Jaccard: 55.01%, 
Epoch: 127, CL Loss: 7.8619, Train Loss: 0.4643, Valid Loss: 0.5521, Test  Loss: 0.8521, Train F1: 70.17%, Train Jaccard: 54.05%, Valid F1: 69.74%, Valid Jaccard: 53.54%, Test F1: 71.09%, Test Jaccard: 55.15%, 
Epoch: 128, CL Loss: 7.8593, Train Loss: 0.4719, Valid Loss: 0.5501, Test  Loss: 0.8542, Train F1: 70.42%, Train Jaccard: 54.34%, Valid F1: 69.96%, Valid Jaccard: 53.80%, Test F1: 71.20%, Test Jaccard: 55.28%, 
Epoch: 129, CL Loss: 7.8606, Train Loss: 0.4709, Valid Loss: 0.5491, Test  Loss: 0.8753, Train F1: 70.58%, Train Jaccard: 54.54%, Valid F1: 70.08%, Valid Jaccard: 53.94%, Test F1: 71.23%, Test Jaccard: 55.31%, 
Epoch: 130, CL Loss: 7.8681, Train Loss: 0.4692, Valid Loss: 0.5474, Test  Loss: 0.8539, Train F1: 70.97%, Train Jaccard: 55.00%, Valid F1: 70.41%, Valid Jaccard: 54.34%, Test F1: 71.49%, Test Jaccard: 55.63%, 
Epoch: 131, CL Loss: 7.8719, Train Loss: 0.4683, Valid Loss: 0.5466, Test  Loss: 0.8530, Train F1: 71.06%, Train Jaccard: 55.11%, Valid F1: 70.41%, Valid Jaccard: 54.33%, Test F1: 71.42%, Test Jaccard: 55.55%, 
Epoch: 132, CL Loss: 7.8622, Train Loss: 0.4665, Valid Loss: 0.5451, Test  Loss: 0.8520, Train F1: 71.32%, Train Jaccard: 55.42%, Valid F1: 70.40%, Valid Jaccard: 54.32%, Test F1: 71.44%, Test Jaccard: 55.56%, 
Epoch: 133, CL Loss: 7.8611, Train Loss: 0.4656, Valid Loss: 0.5442, Test  Loss: 0.8517, Train F1: 71.38%, Train Jaccard: 55.49%, Valid F1: 70.58%, Valid Jaccard: 54.54%, Test F1: 71.52%, Test Jaccard: 55.67%, 
Epoch: 134, CL Loss: 7.8595, Train Loss: 0.4546, Valid Loss: 0.5418, Test  Loss: 0.8510, Train F1: 71.51%, Train Jaccard: 55.66%, Valid F1: 70.79%, Valid Jaccard: 54.79%, Test F1: 71.60%, Test Jaccard: 55.77%, 
Epoch: 135, CL Loss: 7.8569, Train Loss: 0.4538, Valid Loss: 0.5406, Test  Loss: 0.8294, Train F1: 71.58%, Train Jaccard: 55.74%, Valid F1: 70.94%, Valid Jaccard: 54.96%, Test F1: 71.81%, Test Jaccard: 56.02%, 
Epoch: 136, CL Loss: 7.8602, Train Loss: 0.4523, Valid Loss: 0.5388, Test  Loss: 0.8062, Train F1: 71.66%, Train Jaccard: 55.84%, Valid F1: 71.02%, Valid Jaccard: 55.07%, Test F1: 71.81%, Test Jaccard: 56.02%, 
Epoch: 137, CL Loss: 7.8617, Train Loss: 0.4516, Valid Loss: 0.5381, Test  Loss: 0.7994, Train F1: 71.74%, Train Jaccard: 55.93%, Valid F1: 71.06%, Valid Jaccard: 55.11%, Test F1: 71.87%, Test Jaccard: 56.09%, 
Epoch: 138, CL Loss: 7.8560, Train Loss: 0.4501, Valid Loss: 0.5371, Test  Loss: 0.7975, Train F1: 71.88%, Train Jaccard: 56.10%, Valid F1: 70.88%, Valid Jaccard: 54.89%, Test F1: 71.99%, Test Jaccard: 56.24%, 
Epoch: 139, CL Loss: 7.8568, Train Loss: 0.4493, Valid Loss: 0.5366, Test  Loss: 0.7983, Train F1: 71.94%, Train Jaccard: 56.17%, Valid F1: 71.06%, Valid Jaccard: 55.11%, Test F1: 72.18%, Test Jaccard: 56.47%, 
Epoch: 140, CL Loss: 7.8572, Train Loss: 0.4478, Valid Loss: 0.5358, Test  Loss: 0.7970, Train F1: 71.97%, Train Jaccard: 56.21%, Valid F1: 71.34%, Valid Jaccard: 55.44%, Test F1: 72.34%, Test Jaccard: 56.67%, 
Epoch: 141, CL Loss: 7.8557, Train Loss: 0.4471, Valid Loss: 0.5356, Test  Loss: 0.7722, Train F1: 72.06%, Train Jaccard: 56.32%, Valid F1: 71.37%, Valid Jaccard: 55.48%, Test F1: 72.72%, Test Jaccard: 57.13%, 
Epoch: 142, CL Loss: 7.8575, Train Loss: 0.4457, Valid Loss: 0.5358, Test  Loss: 0.7720, Train F1: 72.36%, Train Jaccard: 56.69%, Valid F1: 71.39%, Valid Jaccard: 55.51%, Test F1: 72.69%, Test Jaccard: 57.09%, 
Epoch: 143, CL Loss: 7.8569, Train Loss: 0.4451, Valid Loss: 0.5359, Test  Loss: 0.7719, Train F1: 72.31%, Train Jaccard: 56.63%, Valid F1: 71.38%, Valid Jaccard: 55.50%, Test F1: 72.85%, Test Jaccard: 57.29%, 
Epoch: 144, CL Loss: 7.8616, Train Loss: 0.4437, Valid Loss: 0.5356, Test  Loss: 0.7720, Train F1: 72.57%, Train Jaccard: 56.95%, Valid F1: 71.56%, Valid Jaccard: 55.72%, Test F1: 72.84%, Test Jaccard: 57.28%, 
Epoch: 145, CL Loss: 7.8750, Train Loss: 0.4432, Valid Loss: 0.5353, Test  Loss: 0.7723, Train F1: 72.63%, Train Jaccard: 57.03%, Valid F1: 71.50%, Valid Jaccard: 55.64%, Test F1: 72.65%, Test Jaccard: 57.05%, 
Epoch: 146, CL Loss: 7.8727, Train Loss: 0.4420, Valid Loss: 0.5344, Test  Loss: 0.7732, Train F1: 72.75%, Train Jaccard: 57.17%, Valid F1: 71.44%, Valid Jaccard: 55.56%, Test F1: 72.70%, Test Jaccard: 57.11%, 
Epoch: 147, CL Loss: 7.8570, Train Loss: 0.4414, Valid Loss: 0.5338, Test  Loss: 0.7731, Train F1: 72.78%, Train Jaccard: 57.21%, Valid F1: 71.73%, Valid Jaccard: 55.92%, Test F1: 72.73%, Test Jaccard: 57.14%, 
Epoch: 148, CL Loss: 7.8737, Train Loss: 0.4401, Valid Loss: 0.5327, Test  Loss: 0.7719, Train F1: 72.98%, Train Jaccard: 57.46%, Valid F1: 71.88%, Valid Jaccard: 56.10%, Test F1: 72.96%, Test Jaccard: 57.43%, 
Epoch: 149, CL Loss: 7.8573, Train Loss: 0.4395, Valid Loss: 0.5326, Test  Loss: 0.7716, Train F1: 73.10%, Train Jaccard: 57.61%, Valid F1: 72.06%, Valid Jaccard: 56.32%, Test F1: 73.03%, Test Jaccard: 57.51%, 
Epoch: 150, CL Loss: 7.8580, Train Loss: 0.4383, Valid Loss: 0.5328, Test  Loss: 0.7713, Train F1: 73.31%, Train Jaccard: 57.87%, Valid F1: 72.64%, Valid Jaccard: 57.03%, Test F1: 73.27%, Test Jaccard: 57.81%, 
Epoch: 151, CL Loss: 7.8664, Train Loss: 0.4378, Valid Loss: 0.5324, Test  Loss: 0.7922, Train F1: 73.40%, Train Jaccard: 57.97%, Valid F1: 72.78%, Valid Jaccard: 57.20%, Test F1: 73.37%, Test Jaccard: 57.94%, 
Epoch: 152, CL Loss: 7.8630, Train Loss: 0.4367, Valid Loss: 0.5329, Test  Loss: 0.7886, Train F1: 73.56%, Train Jaccard: 58.17%, Valid F1: 72.98%, Valid Jaccard: 57.45%, Test F1: 73.34%, Test Jaccard: 57.91%, 
Epoch: 153, CL Loss: 7.8616, Train Loss: 0.4361, Valid Loss: 0.5346, Test  Loss: 0.7875, Train F1: 73.67%, Train Jaccard: 58.31%, Valid F1: 73.12%, Valid Jaccard: 57.63%, Test F1: 73.39%, Test Jaccard: 57.96%, 
Epoch: 154, CL Loss: 7.8595, Train Loss: 0.4352, Valid Loss: 0.5619, Test  Loss: 0.7894, Train F1: 73.95%, Train Jaccard: 58.66%, Valid F1: 73.19%, Valid Jaccard: 57.71%, Test F1: 73.40%, Test Jaccard: 57.98%, 
Epoch: 155, CL Loss: 7.8545, Train Loss: 0.4348, Valid Loss: 0.5625, Test  Loss: 0.7901, Train F1: 73.99%, Train Jaccard: 58.72%, Valid F1: 73.27%, Valid Jaccard: 57.82%, Test F1: 73.28%, Test Jaccard: 57.82%, 
Epoch: 156, CL Loss: 7.8536, Train Loss: 0.4336, Valid Loss: 0.5629, Test  Loss: 0.7904, Train F1: 74.06%, Train Jaccard: 58.80%, Valid F1: 73.27%, Valid Jaccard: 57.82%, Test F1: 73.13%, Test Jaccard: 57.64%, 
Epoch: 157, CL Loss: 7.8566, Train Loss: 0.4331, Valid Loss: 0.5657, Test  Loss: 0.7903, Train F1: 74.00%, Train Jaccard: 58.73%, Valid F1: 73.33%, Valid Jaccard: 57.89%, Test F1: 73.20%, Test Jaccard: 57.72%, 
Epoch: 158, CL Loss: 7.8542, Train Loss: 0.4319, Valid Loss: 0.5642, Test  Loss: 0.7896, Train F1: 73.99%, Train Jaccard: 58.72%, Valid F1: 73.59%, Valid Jaccard: 58.21%, Test F1: 73.23%, Test Jaccard: 57.76%, 
Epoch: 159, CL Loss: 7.8588, Train Loss: 0.4313, Valid Loss: 0.5636, Test  Loss: 0.7894, Train F1: 73.91%, Train Jaccard: 58.61%, Valid F1: 73.47%, Valid Jaccard: 58.07%, Test F1: 73.34%, Test Jaccard: 57.90%, 
Epoch: 160, CL Loss: 7.8563, Train Loss: 0.4301, Valid Loss: 0.5623, Test  Loss: 0.7890, Train F1: 73.94%, Train Jaccard: 58.65%, Valid F1: 73.57%, Valid Jaccard: 58.18%, Test F1: 73.39%, Test Jaccard: 57.97%, 
Epoch: 161, CL Loss: 7.8538, Train Loss: 0.4296, Valid Loss: 0.5617, Test  Loss: 0.7886, Train F1: 74.04%, Train Jaccard: 58.78%, Valid F1: 73.59%, Valid Jaccard: 58.21%, Test F1: 73.48%, Test Jaccard: 58.08%, 
Epoch: 162, CL Loss: 7.8568, Train Loss: 0.4284, Valid Loss: 0.5606, Test  Loss: 0.7875, Train F1: 74.14%, Train Jaccard: 58.90%, Valid F1: 73.72%, Valid Jaccard: 58.38%, Test F1: 73.39%, Test Jaccard: 57.97%, 
Epoch: 163, CL Loss: 7.8534, Train Loss: 0.4279, Valid Loss: 0.5601, Test  Loss: 0.7869, Train F1: 74.18%, Train Jaccard: 58.96%, Valid F1: 73.67%, Valid Jaccard: 58.31%, Test F1: 73.57%, Test Jaccard: 58.19%, 
Epoch: 164, CL Loss: 7.8531, Train Loss: 0.4269, Valid Loss: 0.5590, Test  Loss: 0.7867, Train F1: 74.21%, Train Jaccard: 59.00%, Valid F1: 73.69%, Valid Jaccard: 58.34%, Test F1: 73.75%, Test Jaccard: 58.42%, 
Epoch: 165, CL Loss: 7.8519, Train Loss: 0.4264, Valid Loss: 0.5585, Test  Loss: 0.7862, Train F1: 74.24%, Train Jaccard: 59.04%, Valid F1: 73.69%, Valid Jaccard: 58.34%, Test F1: 73.84%, Test Jaccard: 58.53%, 
Epoch: 166, CL Loss: 7.8566, Train Loss: 0.4257, Valid Loss: 0.5577, Test  Loss: 0.8001, Train F1: 74.17%, Train Jaccard: 58.94%, Valid F1: 73.80%, Valid Jaccard: 58.48%, Test F1: 73.76%, Test Jaccard: 58.43%, 
Epoch: 167, CL Loss: 7.8515, Train Loss: 0.4159, Valid Loss: 0.5575, Test  Loss: 0.7635, Train F1: 74.20%, Train Jaccard: 58.98%, Valid F1: 73.84%, Valid Jaccard: 58.52%, Test F1: 73.76%, Test Jaccard: 58.43%, 
Epoch: 168, CL Loss: 7.8582, Train Loss: 0.4149, Valid Loss: 0.5570, Test  Loss: 0.7356, Train F1: 74.23%, Train Jaccard: 59.02%, Valid F1: 73.81%, Valid Jaccard: 58.49%, Test F1: 73.73%, Test Jaccard: 58.39%, 
Epoch: 169, CL Loss: 7.8492, Train Loss: 0.4145, Valid Loss: 0.5568, Test  Loss: 0.7341, Train F1: 74.31%, Train Jaccard: 59.13%, Valid F1: 73.72%, Valid Jaccard: 58.38%, Test F1: 73.63%, Test Jaccard: 58.27%, 
Epoch: 170, CL Loss: 7.8553, Train Loss: 0.4136, Valid Loss: 0.5564, Test  Loss: 0.7330, Train F1: 74.45%, Train Jaccard: 59.30%, Valid F1: 73.75%, Valid Jaccard: 58.41%, Test F1: 73.65%, Test Jaccard: 58.29%, 
Epoch: 171, CL Loss: 7.8521, Train Loss: 0.4132, Valid Loss: 0.5560, Test  Loss: 0.7328, Train F1: 74.47%, Train Jaccard: 59.32%, Valid F1: 73.84%, Valid Jaccard: 58.52%, Test F1: 73.65%, Test Jaccard: 58.29%, 
Epoch: 172, CL Loss: 7.8521, Train Loss: 0.4122, Valid Loss: 0.5553, Test  Loss: 0.7525, Train F1: 74.55%, Train Jaccard: 59.42%, Valid F1: 73.88%, Valid Jaccard: 58.58%, Test F1: 73.84%, Test Jaccard: 58.53%, 
Epoch: 173, CL Loss: 7.8499, Train Loss: 0.4118, Valid Loss: 0.5552, Test  Loss: 0.7517, Train F1: 74.67%, Train Jaccard: 59.57%, Valid F1: 73.91%, Valid Jaccard: 58.61%, Test F1: 73.84%, Test Jaccard: 58.52%, 
Epoch: 174, CL Loss: 7.8492, Train Loss: 0.4109, Valid Loss: 0.5551, Test  Loss: 0.7498, Train F1: 74.91%, Train Jaccard: 59.89%, Valid F1: 73.84%, Valid Jaccard: 58.53%, Test F1: 73.86%, Test Jaccard: 58.56%, 
Epoch: 175, CL Loss: 7.8481, Train Loss: 0.4104, Valid Loss: 0.5552, Test  Loss: 0.7490, Train F1: 75.00%, Train Jaccard: 60.00%, Valid F1: 73.83%, Valid Jaccard: 58.52%, Test F1: 73.83%, Test Jaccard: 58.52%, 
Epoch: 176, CL Loss: 7.8511, Train Loss: 0.4096, Valid Loss: 0.5555, Test  Loss: 0.7469, Train F1: 75.19%, Train Jaccard: 60.24%, Valid F1: 73.98%, Valid Jaccard: 58.70%, Test F1: 74.05%, Test Jaccard: 58.79%, 
Epoch: 177, CL Loss: 7.8470, Train Loss: 0.4093, Valid Loss: 0.5557, Test  Loss: 0.7462, Train F1: 75.14%, Train Jaccard: 60.18%, Valid F1: 74.03%, Valid Jaccard: 58.77%, Test F1: 74.17%, Test Jaccard: 58.94%, 
Epoch: 178, CL Loss: 7.8549, Train Loss: 0.4087, Valid Loss: 0.5559, Test  Loss: 0.7454, Train F1: 75.19%, Train Jaccard: 60.24%, Valid F1: 74.04%, Valid Jaccard: 58.79%, Test F1: 74.18%, Test Jaccard: 58.95%, 
Epoch: 179, CL Loss: 7.8477, Train Loss: 0.4083, Valid Loss: 0.5561, Test  Loss: 0.7450, Train F1: 75.24%, Train Jaccard: 60.31%, Valid F1: 74.10%, Valid Jaccard: 58.86%, Test F1: 74.20%, Test Jaccard: 58.98%, 
Epoch: 180, CL Loss: 7.8594, Train Loss: 0.4077, Valid Loss: 0.5562, Test  Loss: 0.7451, Train F1: 75.24%, Train Jaccard: 60.31%, Valid F1: 74.14%, Valid Jaccard: 58.91%, Test F1: 74.37%, Test Jaccard: 59.20%, 
Epoch: 181, CL Loss: 7.8489, Train Loss: 0.4074, Valid Loss: 0.5560, Test  Loss: 0.7448, Train F1: 75.19%, Train Jaccard: 60.25%, Valid F1: 74.14%, Valid Jaccard: 58.91%, Test F1: 74.43%, Test Jaccard: 59.27%, 
Epoch: 182, CL Loss: 7.8617, Train Loss: 0.4161, Valid Loss: 0.5306, Test  Loss: 0.7438, Train F1: 75.35%, Train Jaccard: 60.45%, Valid F1: 74.17%, Valid Jaccard: 58.94%, Test F1: 74.39%, Test Jaccard: 59.23%, 
Epoch: 183, CL Loss: 7.8537, Train Loss: 0.4158, Valid Loss: 0.5299, Test  Loss: 0.7433, Train F1: 75.34%, Train Jaccard: 60.44%, Valid F1: 74.17%, Valid Jaccard: 58.94%, Test F1: 74.36%, Test Jaccard: 59.19%, 
Epoch: 184, CL Loss: 7.8514, Train Loss: 0.4152, Valid Loss: 0.5274, Test  Loss: 0.7428, Train F1: 75.49%, Train Jaccard: 60.63%, Valid F1: 74.26%, Valid Jaccard: 59.06%, Test F1: 74.44%, Test Jaccard: 59.28%, 
Epoch: 185, CL Loss: 7.8620, Train Loss: 0.4149, Valid Loss: 0.5260, Test  Loss: 0.7427, Train F1: 75.51%, Train Jaccard: 60.65%, Valid F1: 74.40%, Valid Jaccard: 59.24%, Test F1: 74.43%, Test Jaccard: 59.27%, 
Epoch: 186, CL Loss: 7.8509, Train Loss: 0.4139, Valid Loss: 0.5232, Test  Loss: 0.7432, Train F1: 75.54%, Train Jaccard: 60.70%, Valid F1: 74.48%, Valid Jaccard: 59.34%, Test F1: 74.54%, Test Jaccard: 59.41%, 
Epoch: 187, CL Loss: 7.8532, Train Loss: 0.4134, Valid Loss: 0.5222, Test  Loss: 0.7430, Train F1: 75.59%, Train Jaccard: 60.76%, Valid F1: 74.61%, Valid Jaccard: 59.50%, Test F1: 74.59%, Test Jaccard: 59.48%, 
Epoch: 188, CL Loss: 7.8512, Train Loss: 0.4123, Valid Loss: 0.5210, Test  Loss: 0.7431, Train F1: 75.73%, Train Jaccard: 60.94%, Valid F1: 74.61%, Valid Jaccard: 59.50%, Test F1: 74.53%, Test Jaccard: 59.40%, 
Epoch: 189, CL Loss: 7.8473, Train Loss: 0.4025, Valid Loss: 0.5204, Test  Loss: 0.7378, Train F1: 75.88%, Train Jaccard: 61.13%, Valid F1: 74.83%, Valid Jaccard: 59.78%, Test F1: 74.58%, Test Jaccard: 59.47%, 
Epoch: 190, CL Loss: 7.8458, Train Loss: 0.4014, Valid Loss: 0.5193, Test  Loss: 0.7368, Train F1: 75.95%, Train Jaccard: 61.22%, Valid F1: 75.02%, Valid Jaccard: 60.03%, Test F1: 74.62%, Test Jaccard: 59.52%, 
Epoch: 191, CL Loss: 7.8509, Train Loss: 0.4009, Valid Loss: 0.5188, Test  Loss: 0.7366, Train F1: 76.04%, Train Jaccard: 61.34%, Valid F1: 75.17%, Valid Jaccard: 60.22%, Test F1: 74.73%, Test Jaccard: 59.66%, 
Epoch: 192, CL Loss: 7.8480, Train Loss: 0.3999, Valid Loss: 0.5179, Test  Loss: 0.7363, Train F1: 76.17%, Train Jaccard: 61.51%, Valid F1: 75.09%, Valid Jaccard: 60.11%, Test F1: 74.79%, Test Jaccard: 59.73%, 
Epoch: 193, CL Loss: 7.8462, Train Loss: 0.3993, Valid Loss: 0.5174, Test  Loss: 0.7360, Train F1: 76.28%, Train Jaccard: 61.66%, Valid F1: 75.03%, Valid Jaccard: 60.04%, Test F1: 74.87%, Test Jaccard: 59.84%, 
Epoch: 194, CL Loss: 7.8477, Train Loss: 0.3984, Valid Loss: 0.5165, Test  Loss: 0.7159, Train F1: 76.30%, Train Jaccard: 61.68%, Valid F1: 75.00%, Valid Jaccard: 60.00%, Test F1: 74.96%, Test Jaccard: 59.95%, 
Epoch: 195, CL Loss: 7.8468, Train Loss: 0.3980, Valid Loss: 0.5161, Test  Loss: 0.7170, Train F1: 76.30%, Train Jaccard: 61.68%, Valid F1: 74.91%, Valid Jaccard: 59.89%, Test F1: 74.93%, Test Jaccard: 59.90%, 
Epoch: 196, CL Loss: 7.8436, Train Loss: 0.3971, Valid Loss: 0.5153, Test  Loss: 0.7386, Train F1: 76.33%, Train Jaccard: 61.72%, Valid F1: 74.92%, Valid Jaccard: 59.90%, Test F1: 75.14%, Test Jaccard: 60.18%, 
Epoch: 197, CL Loss: 7.8455, Train Loss: 0.3967, Valid Loss: 0.5149, Test  Loss: 0.7391, Train F1: 76.32%, Train Jaccard: 61.71%, Valid F1: 75.03%, Valid Jaccard: 60.04%, Test F1: 75.19%, Test Jaccard: 60.25%, 
Epoch: 198, CL Loss: 7.8462, Train Loss: 0.3960, Valid Loss: 0.5143, Test  Loss: 0.7402, Train F1: 76.45%, Train Jaccard: 61.88%, Valid F1: 75.00%, Valid Jaccard: 60.00%, Test F1: 75.23%, Test Jaccard: 60.30%, 
Epoch: 199, CL Loss: 7.8444, Train Loss: 0.3956, Valid Loss: 0.5141, Test  Loss: 0.7406, Train F1: 76.48%, Train Jaccard: 61.92%, Valid F1: 74.97%, Valid Jaccard: 59.96%, Test F1: 75.29%, Test Jaccard: 60.37%, 
Run 01:
Highest Train F1: 76.48
Highest Train Jaccard: 61.92
Highest Valid F1: 75.17
Highest Valid Jaccard: 60.22
  Final Train F1: 76.04
  Final Train Jaccard: 61.34
   Final Test F1: 74.73
   Final Test Jaccard: 59.66
```
