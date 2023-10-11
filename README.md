# HyGCL-DC

Hypergraph Contrastive Learning for Drug Trafficking Community Detection
====
Official source code of "Hypergraph Contrastive Learning for Drug Trafficking Community Detection" 
(ICDM 2023)

<div>
<img src="https://github.com/GraphResearcher/HyGCL-DC/blob/main/figs/framework.png" width="1200" height="400">
<h5>Fig. 1: The overall framework HyGCL-DC: DC: (a) it first constructs a hypergraph G based on the interactions among online
drug-related users; (b) it integrates augmentations from the structure view and the attribute view to augment hypergraphs into
 ̃G1 and  ̃G2. HyGCL-DC is designed as an end-to-end framework that integrates self-supervised contrastive learning to boost the
node embeddings over unlabeled data by reaching the agreement among positive and negative embedding pairs and supervised
learning with community labels for downstream drug trafficking community detetection</h5>
</div>


## Requirements

This code is developed and tested with python 3.11.1 and the required packages are listed in the `requirements.txt`.

Please run `pip install -r requirements.txt` to install all the dependencies. 


## Twitter-HyDrug
### New Dataset

Twitter-HyDrug is a real-world hypergraph data that describe the drug trafficking communities on Twitter. 
We first crawl the metadata (275,884,694 posts and 40,780,721 users) through the official
Twitter API from Dec 2020 to Aug 2021. 
Afterward, we generate a drug keyword list that covers 21 drug types that may cause drug overdose or drug addiction 
problems to filter the tweets that contain drug-relevant information. Based on the keyword list, we obtain 266,975
filtered drug-relevant posts by 54,680 users.
Moreover, we define six type of drug communities, i.e., cannabis, opioid, hallucinogen, stimulant, depressant, and
others communities, based on the drug functions.
Six researchers spent 62 days annotating these Twitter users into six communities based on the annotation rules discussed in next section.
With the specific criteria, six researchers annotate the filtered metadata sperately. For these Twitter users with disagreed labels, 
we conduct further discussion among annotators for cross-validation. To conclude, we obtain the drug hypergraph including 2,725 user nodes and 35,012 hyperedges.

### Content Feature
For each Twitter user, we consider as a node in Twitter-HyDrug. Then, for each Twitter user, we extract the text information, including
usernames, profiles, and tweets to describe these users within drug communities. To further accurately characterize Twitter
users, we categorize the tweets into two groups: drug-related and drug-unrelated tweets, with a set of drug keywords. 
Then, as most users have sufficient tweets on Twitter, we select partial tweets including all drug-related
tweets and five drug-unrelated tweets to effectively and efficiently represent these users. Specifically, for drug-unrelated
tweets, we prioritize drug-unrelated tweets involving active interactions with other users in the drug community. For
instance, if the number of drug-unrelated tweets is much more than five, then we select these tweets with more interactions.
On the contrary, if the user has fewer than five drug-unrelated tweets, we leverage all the drug-unrelated tweets. Then
we combine profiles, usernames, drug-related tweets, as well as drug-unrelated tweets, and further feed them to the pre-
trained transformer-based language model, SentenceBert, to obtain fixed-length feature vectors.

### Hyperedge
To exhaustively depict the complex and group- wise relationships among users in Twitter-HyDrug, we define
four types of hyperedges for describing the activities among users as follows: (i) R1: users-follow-user hyperedge 
relation denotes that a group of users follow a specific user in Twitter-HyDrug. The follow/following-based hyperedge 
aims to represent the social connections within drug trafficking communities, illustrating the friend circles involved 
in such illicit activities. (ii) R2: users-engage-conversation hyperedge relation represents that a group of users is 
engaged in a tweet-based conversation, encompassing activities such as posting, replying, retweeting, and liking the 
tweets involved within the conversation. The conversation-based hyperedge serves to portray the shared interests and 
topics among the group of users. (iii) R3: users-include-hashtag hyperedge relation
indicates that a bunch of users actively discuss the specific
hashtag-based topics by posting the specific hashtag in tweets
or profiles. Partial hashtag keywords are listed in TABLE I. For
instance, a group of users posts tweets on Twitter that include
oxycodone, one of the opioid drugs. Then the oxycodone hashtag will be considered as a hyperedge that encompasses all of
the users in this group. (iv) R4: users-contain-emoji hyperedge relation signifies that a bunch of users contains a specific drug-related emoji in their tweets or profiles. 
Examples of drug-related emojis are illustrated in Fig. 2(b). Similar to hashtags,
we use emojis to describe the interested drugs in this group.

### Annotation Rules
(i) If a user actively promotes some type of drug on Twitter or has rich connections (e.g.,
following, replying, liking, and retweeting) with other drug- related users in specific drug communities, he/she will be 
considered a member of the corresponding drug communities. For instance, Fig. 2(b) shows a drug seller who advertises his/her
drugs including oxycodone, cocaine, and xanax. Based on the function of these drugs, we consider this user a member of
the overlapping communities including the opioid community (oxycodone), stimulant community (cocaine), and depressant
community (Xanax). (ii) If a user appears to suffer from drug overdose or drug addiction to the specific drug, he/she will
be regarded as a member of the specific drug community. For instance, Fig. 2(c) illustrates a drug user ”Ua\*\*\*on” that
suffers from an opioid overdose. So we classify this user as a member of the opioid community. (iii) If we can find evidence
on Twitter that a user used to purchase specific drugs from others on Twitter, then he/she belongs to a member of the
corresponding drug community. For example, Fig. 2(c) shows a drug buyer ”Bd\*\*\*in” that purchased oxycodone from drug
seller ”Su\*\*\*oy”. We consider he/she as a member of the opioid community. (iv) If we could not find any evidence that
a user suffers from drug overdoses or purchases from others, instead, he/she is very actively involved in discussing and
propagating specific drugs, we also consider he/she a member of the corresponding drug community. For instance, if a user
actively retweets LSD promotion tweets, but does not show evidence of purchasing or having LSD on Twitter, he/she is
still regarded as a member of the hallucinogen community.(v) If a user is promoting or using or purchasing or discussing
drugs on Twitter, but he/she does not mention the specific type of drugs, then we consider the user a member of others drug
community. For instance, if a user complains that he/she is suffering from drug overdoses but does not mention the type
of drugs that he/she is addicted to, then we regard the user as a member of others community. Mention that, partial of drugs
and the corresponding communities are listed in TABLE I. Based on the above strategy, we can obtain the ground truth
for the drug trafficking community detection task.

<div align="center">
<img src="https://github.com/GraphResearcher/HyGCL-DC/blob/main/figs/Intro.jpg" width="500" height="500">
<h5>Fig. 2: Illustration about drug trafficking communities among users on Twitter</h5>
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