# Beat Tracking and Tempo Estimation Across Musical Cultures (DP2_B)

CSC 475/575 Music Information Retrieval, University of Victoria, Fall 2026

Instructor: George Tzanetakis

Group members:

- Irai Kumaran Sivanesan (iraikumarans@uvic.ca)
- Nalluraj Babu (nallurajbabu@uvic.ca)
- Usha Sundaresan Manivannan (mvannan@uvic.ca)

Repository: [add GitHub link here]

---

## 1. Project Overview

When people listen to music, they often tap their foot along with it. Two things happen at the same time. They sense how fast the music is, which is the tempo in beats per minute (BPM). They also place each tap at the right moment, which is the beat position. People do this without effort, but it is hard for a computer. To a computer, a song is only a long list of numbers that describe air pressure over time. Nothing in that list says where the beat is.

This project covers two related tasks:

- **Tempo estimation**: given an audio clip, output one global tempo value in BPM.
- **Beat tracking**: given an audio clip, output a list of time stamps (in seconds) where the beats occur.

We will build three different methods from scratch and compare them with established systems on Western music. This is the core of the project and follows Default Project 2. We also propose an extension that tests all of these methods on South Indian music. Because the extension goes beyond the resources provided for DP2, Section 11 describes it in detail and lists the questions we would like the instructor's feedback on before we commit to it.

## 2. Problem Statement

Most beat tracking research has been developed and tested on Western popular music. In that music the beat is usually steady and each bar has four beats. The standard datasets, such as GTZAN, reflect this. As a result, we know a lot about how well beat trackers work on Western pop and rock, and much less about how they work on other kinds of music.

South Indian music is a useful test case because it is organised differently:

- **Carnatic music** is the classical tradition of South India. Its rhythm follows a tala, which is a repeating cycle of beats. Common talas have 8, 3, 7 or 5 beats per cycle, and some of these cycles are split into uneven groups (for example 3+2+2). The first beat of each cycle is called the sama. The tempo can drift during a performance, and the mridangam (the main percussion instrument) often plays dense patterns between the beats.
- **Tamil film music** is popular music from the Tamil film industry. It often mixes Western drum kits and electronic beats with Indian percussion and rhythmic patterns. It sits between Western pop and Carnatic music. As far as we know, there is no public dataset of Tamil film songs with beat annotations.

Our central question is: **do tempo and beat tracking methods built for Western music still work on South Indian music, and if not, where and why do they fail?**

## 3. Motivation

Beat and tempo information is used in many real applications, including DJ software, playlist generation, music games, automatic accompaniment and fitness apps that match music to running pace. It is also a first step for other tasks such as downbeat tracking, chord recognition and structure analysis. If these methods only work well on Western music, every application built on them inherits the same limitation.

Tamil film music is listened to by tens of millions of people, yet it is almost absent from MIR datasets. Carnatic music has been studied by a small number of research groups, but mostly with specialised models designed for it. Testing the common, general purpose tools on this music gives a clear picture of how far they can be trusted outside their usual setting.

The project also has a learning goal. Each member will implement one complete method from scratch, so that we understand how these systems work rather than only calling library functions.

## 4. Project Vision and Expected Outcomes

At the end of the term we expect to deliver the following:

1. **Three working methods built from scratch**: an autocorrelation tempo estimator, a dynamic programming beat tracker and a neural network beat tracker trained by us.
2. **A benchmark on Western music**: accuracy and speed of our methods compared with librosa, madmom and the 1D State Space tracker on GTZAN and GiantSteps.
3. **A new annotated test set**: about 40 Tamil film music clips with beat annotations created by our group, released in the repository with the song list (annotations only, not audio).
4. **A cross-cultural evaluation**: results of all methods on Tamil film music and Carnatic music, broken down by style and by tala.
5. **An analysis of failures**: which kinds of errors appear (for example half or double tempo, off-beat tracking, losing the beat when the tempo drifts) and why.
6. **A sama finding experiment**: a test of whether the first beat of each tala cycle can be found when the tracker is told the length of the cycle.
7. **A final report** in ISMIR paper format, and a public repository with code, results and instructions to reproduce them.

Our hypotheses, which we will test, are:

- All methods will perform best on Western music, worse on Tamil film music, and worst on Carnatic music.
- On Carnatic music, the talas with odd cycle lengths (Misra Chapu and Khanda Chapu) will cause more errors than Adi tala.
- Neural network trackers trained on Western data will lose more accuracy on Carnatic music than the simpler signal processing methods, because they have learned patterns that are specific to Western music.

## 5. Datasets

| Dataset | Content | Annotations | Access | Used for |
|---|---|---|---|---|
| GTZAN | 1000 clips of 30 s in 10 genres | Beats and tempo (gtzan_tempo_beat) | Audio from the instructor, annotations on GitHub | All three methods |
| GiantSteps Tempo | 664 electronic dance music previews | Tempo | Download script on GitHub | Tempo estimation |
| CompMusic Carnatic Music Rhythm (CMR) | Carnatic recordings in Adi, Rupaka, Misra Chapu and Khanda Chapu tala, including a subset of 118 two minute excerpts | Beats and sama | On request through Zenodo for non-commercial research | Carnatic beat and sama tracking |
| Saraga Carnatic | 249 Carnatic recordings | Sama and tempo | Open, through mirdata | Backup if CMR access is delayed |
| Tamil film music (new) | About 40 clips of 30 s from several decades | Beats, created by us | Annotations and song list released in our repository | Tamil film evaluation |

Links:

- GTZAN annotations: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
- GiantSteps Tempo: https://github.com/GiantSteps/giantsteps-tempo-dataset
- CMR: https://zenodo.org/records/1264394

**Creating the Tamil film music test set.** We will select about 40 songs from different decades and with different rhythmic styles. For each song we take one 30 second clip. We will annotate the beats by tapping along while the clip plays, and then correct each tap by looking at the waveform in Sonic Visualiser [25]. To check that the annotations are reliable, 10 clips will be annotated a second time on a different day and the two versions will be compared with the same metrics we use for the trackers. The annotation files will use the same format as the GTZAN annotations, so that all evaluation code works on both.

We will use fixed train and test splits wherever a method needs training, and report all results on the same test tracks.

## 6. Methodology

### 6.1 Shared pipeline

All three members share a small amount of common code:

1. A data loading module that reads audio and annotations for every dataset in the same format.
2. An evaluation module that wraps mir_eval and saves results as a table with the same columns for every method (track, dataset, genre or tala, method, metric, value).

Everything else is owned by one member. This keeps the individual objectives independent.

### 6.2 Onset strength envelope

All three methods start from an onset strength envelope. This is a signal that rises sharply when a new note or drum hit starts. We compute it from the change in a log magnitude spectrogram over time. The envelope turns the beat problem into a simpler one: find a regular pattern of peaks.

### 6.3 Tempo estimation by autocorrelation (Irai)

Autocorrelation compares the onset envelope with a shifted copy of itself. If the music repeats every half second, the two copies line up well at a shift of half a second, which corresponds to 120 BPM. The tempo estimate is the shift with the strongest match within a sensible range of 40 to 250 BPM. The main weakness is that a song at 120 BPM also matches well at 60 and 240 BPM. These half and double tempo mistakes are called octave errors. Following Percival and Tzanetakis [4], we add a second stage that cross-correlates the envelope with ideal pulse trains at each candidate tempo, which helps choose between these candidates.

### 6.4 Beat tracking by dynamic programming (Nalluraj)

Following Ellis [5], beat tracking is treated as a search problem. The tracker looks for the sequence of beat times that lands on strong onsets while keeping the spacing between beats close to the expected beat period. Dynamic programming solves this efficiently by keeping, for every point in time, the best score of any beat sequence that ends there. A tightness parameter controls how strictly the spacing must follow the tempo. This method assumes a steady tempo, which makes it a good test case for Carnatic music, where the tempo can drift.

### 6.5 Neural network beat tracking (Usha)

Instead of hand written rules, a neural network learns what a beat sounds like from annotated examples. We will train a temporal convolutional network (TCN) [9] that takes spectrogram frames as input and outputs the probability of a beat at each frame. Beat times are then picked from the peaks of this output. The network will be trained on a training split of GTZAN and tested on held out tracks, and later on the South Indian data.

### 6.6 Reference systems

We compare our methods with established implementations: librosa (tempo and beat tracking), madmom (RNN beat tracker with DBN post-processing, and its joint beat and downbeat tracker) and the 1D State Space tracker [11]. These represent common practice and the recent state of the art.

### 6.7 Sama finding

For Carnatic music, finding the sama is similar to finding the downbeat (the first beat of the bar) in Western music. We will run madmom's downbeat tracker first with its default Western setting of 3 or 4 beats per bar, and then with the correct number of beats for each tala. We will also build a simple sama finder that groups the tracked beats into cycles of the known length and chooses the starting beat whose position has the strongest onsets on average.

### 6.8 Listening checks

Throughout the project we will create click tracks, which are the original audio with a short tick added at every predicted beat. Listening to these is the fastest way to catch bugs and to understand failures that numbers alone do not explain.

## 7. Tools

- Python 3, NumPy, SciPy, Matplotlib, pandas, Jupyter
- librosa: audio loading, spectrograms, onset strength, reference tempo and beat tracking
- madmom: reference beat, downbeat and tempo trackers
- 1D State Space tracker: https://github.com/mjhydri/1D-StateSpace
- PyTorch: training the TCN beat tracker (with Google Colab for GPU access if needed)
- mir_eval: evaluation metrics
- mirdata: dataset loaders
- Sonic Visualiser: beat annotation for the Tamil film music test set
- GitHub: version control, issue tracking and collaboration

## 8. Evaluation Metrics

- **Tempo**: Accuracy1 (estimate within 4% of the annotated tempo) and Accuracy2 (the same, but also accepting double, triple, half and one third of the annotated tempo). The gap between the two shows how many errors are octave errors.
- **Beat tracking**: F-measure with a 70 ms tolerance window, CMLt and AMLt (which check whether the tracker stays correct over continuous stretches) and Information Gain, all computed with mir_eval [17, 19].
- **Sama tracking**: F-measure on the predicted sama positions.
- **Annotation reliability**: the same beat metrics, computed between two annotations of the same clip.
- **Efficiency**: processing time per minute of audio, measured on the same machine.

All results will be reported per dataset, per genre for GTZAN, per style for Tamil film music, and per tala for Carnatic music.

## 9. Timeline

| Period | Milestone |
|---|---|
| Oct 19 to Oct 25 | Design specification, repository setup, CMR access request, shared loading and evaluation code |
| Oct 26 to Nov 8 | Basic performance indicators complete for all members, Tamil film clips selected |
| Nov 9 to Nov 22 | Expected performance indicators in progress, Tamil film annotation complete, first combined results table |
| Progress report (date TBC) | README updated with first results |
| Nov 23 to Dec 6 | Remaining expected indicators, advanced indicators where time allows, error analysis |
| Final deadline (date TBC) | Final report in ISMIR LaTeX format and presentation |

The dates on each performance indicator below follow this timeline. Basic and expected indicators form the core plan. Advanced indicators are stretch goals that we will pursue once the core work is complete.

## 10. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| CMR dataset access is delayed | Request access in the first week. Use the open Saraga dataset for sama and tempo in the meantime. |
| GiantSteps audio links no longer work | Report tempo results on GTZAN and the South Indian data only, and note this in the report. |
| madmom fails to install on recent Python versions | Install from the GitHub source, or use a separate Python 3.9 environment. |
| Annotating the Tamil film clips takes longer than planned | Reduce the set to 25 clips while keeping the 10 clip reliability check. |
| TCN training is slow on a laptop | Train on Google Colab. GTZAN is small enough for a free GPU. |

## 11. Proposed Extension and Questions for the Instructor

### 11.1 What we propose

The core of our project (Objective 1 for each member) follows DP2 and uses Western music only. As an extension (Objective 2 for each member), we propose testing the same methods on two kinds of South Indian music:

- **Carnatic music**, using existing datasets with beat and sama annotations.
- **Tamil film music**, using a small test set that we annotate ourselves, since no public annotated dataset seems to exist.

This would let us study how methods built for Western music behave on music with different rhythmic structure. We think this makes the project more useful and more original than running the standard DP2 pipeline alone. Since it goes beyond the provided resources, we would like to confirm with you that it is feasible before we commit.

### 11.2 How we plan to do it

1. **Week 1:** request access to the CompMusic Carnatic Rhythm dataset on Zenodo. Load the open Saraga dataset at the same time, so work can start even if access is delayed.
2. **Weeks 2 to 4:** select about 40 Tamil film clips and annotate them. At roughly 15 minutes per clip, this is about 10 hours of work. A second annotation of 10 clips will check reliability.
3. **Weeks 3 to 6:** run the methods from Objective 1 on the South Indian data. Most of the code is reused, since the same loaders and evaluation module handle every dataset.
4. **Weeks 5 to 7:** analyse the errors by tala and by rhythm style, and prepare figures and audio examples.

### 11.3 How you could help

- Point us to Carnatic or other Indian music datasets with beat or tempo annotations, or share a local copy if you have one.
- Suggest literature on rhythm analysis of Indian music, or of popular music outside the Western tradition.
- Give feedback on our annotation plan for the Tamil film clips.
- Point us to the Marsyas implementation of your tempo estimator [4], so that Irai can check his implementation against it.

### 11.4 Questions on feasibility

1. **Time:** is the extension realistic within the term alongside the core DP2 work? Would you prefer that we reduce it, for example to Carnatic music only or Tamil film music only?
2. **Datasets:** is the CompMusic Carnatic Rhythm dataset a good choice, and is it acceptable to use it for a course project? Is there a better dataset you would recommend?
3. **Annotation:** is a self-annotated test set of about 40 clips, with a reliability check, acceptable as evaluation data? We plan to publish only the annotations and the song list, not the audio. Is that appropriate?
4. **Scope and grading:** would this extension be seen as going beyond a default project? Should we list it as a student proposal instead of DP2?
5. **Workload:** each member has one build objective and one extension objective. Does this split look balanced and independent enough to you?

### 11.5 Fallback plan

If the extension is not feasible, each member's Objective 2 would be replaced with a Western-only objective of similar size:

- **Irai:** octave error analysis of tempo estimation across GTZAN genres and GiantSteps, including a correction step based on the pulse train stage.
- **Nalluraj:** robustness of beat tracking to degraded audio (added noise, low pass filtering and lossy compression).
- **Usha:** downbeat tracking on GTZAN, comparing madmom with a simple downbeat finder she builds.

---

## 12. Individual Objectives

Each member has one objective focused on building a method from scratch (core DP2) and one objective focused on a part of the South Indian evaluation (proposed extension, see Section 11). The objectives are independent of each other.

## Irai Kumaran Sivanesan

Role: tempo estimation and the Tamil film music test set

* Objective 1 (implement an autocorrelation based tempo estimator from scratch and evaluate it on GTZAN and GiantSteps)
   * PI1 (basic): compute an onset strength envelope and its autocorrelation, and select the strongest peak between 40 and 250 BPM as the tempo estimate (Nov 1)
   * PI2 (basic): report Accuracy1 and Accuracy2 on GTZAN and GiantSteps and count the half and double tempo errors (Nov 8)
   * PI3 (expected): add the pulse train cross-correlation stage described by Percival and Tzanetakis [4] and measure how much it improves accuracy (Nov 15)
   * PI4 (expected): compare against librosa and madmom tempo estimates on the same tracks, with a per-genre results table for GTZAN and timing for each method (Nov 22)
   * PI5 (advanced): replace the single onset envelope with several frequency bands, combine their autocorrelations, and test whether this reduces octave errors (Dec 6)

* Objective 2 (create and evaluate a beat annotated Tamil film music test set)
   * PI1 (basic): select about 40 clips of 30 seconds from Tamil film songs across several decades and record the song list and clip times (Nov 1)
   * PI2 (basic): annotate the beats of every clip by tapping and correcting the taps in Sonic Visualiser, and export them in the same format as the GTZAN annotations (Nov 15)
   * PI3 (expected): check annotation reliability by annotating 10 clips a second time on a different day and scoring the two versions against each other with mir_eval (Nov 22)
   * PI4 (expected): run the from-scratch tempo estimator, librosa and madmom on the clips and compare the results with GTZAN (Nov 29)
   * PI5 (advanced): label each clip by rhythm style (Western kit, Indian percussion or mixed) and test whether accuracy depends on the style (Dec 6)

## Nalluraj Babu

Role: dynamic programming beat tracking and Carnatic beat evaluation

* Objective 1 (implement a dynamic programming beat tracker from scratch and evaluate it on GTZAN)
   * PI1 (basic): compute an onset strength envelope and a global tempo estimate as input to the tracker (Nov 1)
   * PI2 (basic): implement the dynamic programming recursion and backtracking described by Ellis [5] to produce beat times (Nov 8)
   * PI3 (expected): evaluate the tracker with mir_eval (F-measure, CMLt and AMLt) on GTZAN, broken down by genre (Nov 15)
   * PI4 (expected): compare against librosa beat_track and study how the tightness parameter affects the results (Nov 22)
   * PI5 (advanced): extend the tracker so it can follow a slowly changing tempo and test it on tracks with tempo variation (Dec 6)

* Objective 2 (evaluate beat tracking on Carnatic music)
   * PI1 (basic): load the Carnatic excerpts and their beat annotations and check them with click tracks (Nov 8)
   * PI2 (basic): run the dynamic programming tracker, librosa and the madmom beat tracker on all excerpts and report results for each tala (Nov 15)
   * PI3 (expected): find the best tightness setting for Carnatic music and compare it with the best setting for GTZAN (Nov 22)
   * PI4 (expected): compare onset envelopes of Carnatic and GTZAN tracks and test whether a different onset method (for example spectral flux on a low frequency band) improves tracking (Nov 29)
   * PI5 (advanced): measure how much the tempo drifts within each Carnatic excerpt and test whether the trackers fail more often on excerpts with more drift (Dec 6)

## Usha Sundaresan Manivannan

Role: neural network beat tracking and Carnatic sama tracking

* Objective 1 (train a neural network beat tracker from scratch and compare it with pretrained systems)
   * PI1 (basic): build a training pipeline that turns GTZAN audio into spectrogram frames with beat labels, using a fixed train and test split (Nov 1)
   * PI2 (basic): implement and train a small temporal convolutional network (TCN) beat tracker in PyTorch, following Davies and Böck [9] (Nov 8)
   * PI3 (expected): turn the network output into beat times with peak picking and evaluate it on the GTZAN test split with mir_eval (Nov 15)
   * PI4 (expected): compare the network with madmom and the 1D State Space tracker in accuracy and processing time per minute of audio (Nov 22)
   * PI5 (advanced): add data augmentation by time stretching and test whether it improves accuracy on tracks with tempos that are rare in the training data (Dec 6)

* Objective 2 (find the sama of the tala cycle in Carnatic music)
   * PI1 (basic): load the sama annotations and run madmom's downbeat tracker with its default bar lengths of 3 and 4 beats (Nov 8)
   * PI2 (basic): evaluate the predicted downbeats against the annotated sama positions (Nov 15)
   * PI3 (expected): rerun the tracker with the correct number of beats per cycle for each tala (8, 3, 7 or 5) and compare with the default setting (Nov 22)
   * PI4 (expected): implement a simple sama finder that groups the beats into cycles of the known length and picks the starting beat whose position has the strongest onsets, then compare it with madmom (Nov 29)
   * PI5 (advanced): fine-tune the TCN from Objective 1 with an extra sama output on a Carnatic training split and evaluate it on held-out excerpts (Dec 6)

---

## 13. Related Work

Early work on tempo and beat analysis used banks of resonating filters on the changes in audio energy. Scheirer [1] showed that a small set of comb filters could track the beat across many kinds of music. Gouyon et al. [2] ran the first large comparison of tempo induction algorithms. That study set Accuracy1 and Accuracy2 as standard metrics and showed that choosing the wrong metrical level (half or double tempo) is the most common error.

Most classic systems first compute an onset detection function. Bello et al. [3] give a tutorial on these functions. Tempo is then found from repeating patterns in this signal. Percival and Tzanetakis [4] proposed a streamlined estimator based on autocorrelation and cross-correlation with pulse trains, which performed close to much more complex systems. Ellis [5] treated beat tracking as dynamic programming. The method searches for the beat sequence that best matches the onsets while keeping a steady spacing between beats.

Machine learning has since become the dominant approach. Böck and Schedl [6] used recurrent neural networks to predict where beats occur. Krebs et al. [7] proposed an efficient state space model for tracking tempo and meter together. Böck et al. [8] extended this to joint beat and downbeat tracking. Davies and Böck [9] showed that temporal convolutional networks track beats accurately with fewer parameters, and Böck et al. [10] trained tempo and beat together so that each task helps the other. Heydari et al. [11] proposed a one dimensional state space that makes inference much faster. Davies et al. [12] give a practical overview of the field in their ISMIR tutorial, which is also the source of the GTZAN beat annotations we use.

Only a small part of this work looks beyond Western music. Holzapfel et al. [13] showed that a meter tracking model can handle the "odd" meters found in Turkish, Indian and Greek music when it is trained on that music. Srinivasamurthy et al. [14] compared rhythm analysis methods on Turkish and Indian art music and found that methods built for Western music struggle with long and irregular cycles. Srinivasamurthy and Serra [15] proposed a supervised method for tracking tala cycles in Carnatic music, and Srinivasamurthy et al. [16] made this kind of tracking much faster with particle filters. These studies use specialised Bayesian models on classical music. Our project differs in two ways. We test the general purpose tools that most people use today on the same Carnatic data and compare them directly with their behaviour on Western data. We also add Tamil film music, a popular style that mixes both traditions and has no public beat annotations.

For evaluation, Davies et al. [17] describe the beat tracking metrics we use, and Raffel et al. [19] provide their reference implementation in mir_eval. Our Western datasets are GTZAN, introduced by Tzanetakis and Cook [20], and GiantSteps Tempo by Knees et al. [21]. Sturm [22] documents known problems with GTZAN, which we keep in mind when reading per-genre results. Our Carnatic data comes from the CompMusic project [15] and the Saraga collection [18]. The software we use includes librosa [23], madmom [24] and Sonic Visualiser [25].

## 14. References

[1] E. D. Scheirer, "Tempo and beat analysis of acoustic musical signals," Journal of the Acoustical Society of America, vol. 103, no. 1, pp. 588-601, 1998.

[2] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, "An experimental comparison of audio tempo induction algorithms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832-1844, 2006.

[3] J. P. Bello, L. Daudet, S. Abdallah, C. Duxbury, M. Davies, and M. Sandler, "A tutorial on onset detection in music signals," IEEE Transactions on Speech and Audio Processing, vol. 13, no. 5, pp. 1035-1047, 2005.

[4] G. Percival and G. Tzanetakis, "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 1765-1776, 2014.

[5] D. P. W. Ellis, "Beat tracking by dynamic programming," Journal of New Music Research, vol. 36, no. 1, pp. 51-60, 2007.

[6] S. Böck and M. Schedl, "Enhanced beat tracking with context-aware neural networks," in Proc. International Conference on Digital Audio Effects (DAFx), 2011.

[7] F. Krebs, S. Böck, and G. Widmer, "An efficient state-space model for joint tempo and meter tracking," in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2015.

[8] S. Böck, F. Krebs, and G. Widmer, "Joint beat and downbeat tracking with recurrent neural networks," in Proc. ISMIR, 2016.

[9] M. E. P. Davies and S. Böck, "Temporal convolutional networks for musical audio beat tracking," in Proc. European Signal Processing Conference (EUSIPCO), 2019.

[10] S. Böck, M. E. P. Davies, and P. Knees, "Multi-task learning of tempo and beat: Learning one to improve the other," in Proc. ISMIR, 2019.

[11] M. Heydari, M. McCallum, A. Ehmann, and Z. Duan, "A novel 1D state space for efficient music rhythmic analysis," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[12] M. E. P. Davies, S. Böck, and M. Fuentes, "Tempo, beat and downbeat estimation," Tutorial at ISMIR, 2021. [Online]. Available: https://tempobeatdownbeat.github.io/tutorial

[13] A. Holzapfel, F. Krebs, and A. Srinivasamurthy, "Tracking the 'odd': Meter inference in a culturally diverse music corpus," in Proc. ISMIR, 2014.

[14] A. Srinivasamurthy, A. Holzapfel, and X. Serra, "In search of automatic rhythm analysis methods for Turkish and Indian art music," Journal of New Music Research, vol. 43, no. 1, pp. 94-114, 2014.

[15] A. Srinivasamurthy and X. Serra, "A supervised approach to hierarchical metrical cycle tracking from audio music recordings," in Proc. IEEE ICASSP, 2014.

[16] A. Srinivasamurthy, A. Holzapfel, A. T. Cemgil, and X. Serra, "Particle filters for efficient meter tracking with dynamic Bayesian networks," in Proc. ISMIR, pp. 197-203, 2015.

[17] M. E. P. Davies, N. Degara, and M. D. Plumbley, "Evaluation methods for musical audio beat tracking algorithms," Technical Report C4DM-TR-09-06, Queen Mary University of London, 2009.

[18] A. Srinivasamurthy, S. Gulati, R. Caro Repetto, and X. Serra, "Saraga: Open datasets for research on Indian art music," Empirical Musicology Review, vol. 16, no. 1, 2021.

[19] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, "mir_eval: A transparent implementation of common MIR metrics," in Proc. ISMIR, 2014.

[20] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, 2002.

[21] P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. Le Goff, "Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections," in Proc. ISMIR, 2015.

[22] B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use," arXiv preprint arXiv:1306.1461, 2013.

[23] B. McFee, C. Raffel, D. Liang, D. P. W. Ellis, M. McVicar, E. Battenberg, and O. Nieto, "librosa: Audio and music signal analysis in Python," in Proc. Python in Science Conference (SciPy), 2015.

[24] S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer, "madmom: A new Python audio and music signal processing library," in Proc. ACM International Conference on Multimedia, pp. 1174-1178, 2016.

[25] C. Cannam, C. Landone, and M. Sandler, "Sonic Visualiser: An open source application for viewing, analysing, and annotating music audio files," in Proc. ACM International Conference on Multimedia, pp. 1467-1468, 2010.
