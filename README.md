# Beat Tracking and Tempo Estimation Across Musical Cultures (DP2_B)

CSC 475/575 Music Information Retrieval, University of Victoria, Fall 2026

Instructor: George Tzanetakis

Group members:

- Irai Kumaran Sivanesan (iraikumarans@uvic.ca)
- Nalluraj Babu (nallurajbabu@uvic.ca)
- Usha Sundaresan Manivannan (mvannan@uvic.ca)

Repository: https://github.com/Iraikumaran/beat-tracking

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

Tamil film music is widely listened to, yet it is almost absent from MIR datasets. Carnatic music has been studied by a small number of research groups, but mostly with specialised models designed for it. Testing the common, general purpose tools on this music gives a clear picture of how far they can be trusted outside their usual setting.

The project also has a learning goal. Each member will implement one complete method from scratch, so that we understand how these systems work rather than only calling library functions.

## 4. Project Vision and Expected Outcomes

At the end of the term we expect to deliver the following:

1. **Three working methods built from scratch**: an autocorrelation tempo estimator, a dynamic programming beat tracker and a neural network beat tracker trained by us.
2. **A benchmark on Western music**: accuracy and speed of our methods compared with librosa, madmom and the 1D State Space tracker on GTZAN and GiantSteps.
3. **A new annotated test set**: 30 to 40 Tamil film music clips with beat annotations created by our group, released in the repository with the song list (annotations only, not audio).
4. **A cross-cultural evaluation**: results on Tamil film music and Carnatic music, broken down by style and by tala.
5. **An analysis of failures**: which kinds of errors appear (for example half or double tempo, off-beat tracking, losing the beat when the tempo drifts) and why.
6. **A sama finding experiment**: a test of whether the first beat of each tala cycle can be found when the tracker is told the length of the cycle.
7. **A final report** in ISMIR paper format, and a public repository with code, results and instructions to reproduce them.

Our hypotheses, which we will test, are:

- **H1:** All methods will perform best on Western music, worse on Tamil film music, and worst on Carnatic music.
- **H2:** On Carnatic music, the talas with uneven groupings of beats inside the cycle (Misra Chapu and Khanda Chapu) will cause more errors than Adi tala.
- **H3:** Neural network trackers trained on Western data will lose more accuracy on Carnatic music than the signal processing trackers, because they have learned patterns that are specific to Western music. This is tested mainly by comparing madmom with our dynamic programming tracker and librosa. Our own TCN, which is trained only on GTZAN, gives a secondary result.

## 5. Datasets

| Dataset | Content | Annotations | Access | Used for |
|---|---|---|---|---|
| GTZAN | 1000 clips of 30 s in 10 genres | Beats and tempo [1] | Audio from the instructor, annotations from the gtzan_tempo_beat repository | All three methods |
| GiantSteps Tempo | 664 electronic dance music previews of about 2 minutes | Tempo, version 2 annotations | Audio fetched with the repository's download script from Beatport preview links | Tempo estimation |
| CompMusic Carnatic Music Rhythm (CMR) | Carnatic recordings in Adi, Rupaka, Misra Chapu and Khanda Chapu tala, including a subset of 118 two minute excerpts | Beats and sama | On request through Zenodo, non-commercial research only | Carnatic beat and sama tracking |
| Saraga Carnatic | 249 Carnatic recordings | Sama and tempo (no beat annotations) | Open, through mirdata | Fallback for CMR |
| Tamil film music (new) | 30 to 40 clips of 30 s from several decades | Beats, created by us | Annotations and song list in our repository | Tamil film evaluation |

Links:

- GTZAN annotations: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
- GiantSteps Tempo: https://github.com/GiantSteps/giantsteps-tempo-dataset
- CMR: https://zenodo.org/records/1264394

**GTZAN split.** GTZAN contains repeated songs and several clips from the same artists [2]. To avoid results that are too optimistic, all GTZAN results are reported on the artist-filtered, fault-filtered train and test split of Kereliuk et al. [3]. The split is committed as `data/splits/gtzan_split.csv`.

**Carnatic beat level.** In Carnatic music a tala can be performed at more than one speed level (kalai). We will check by ear which level the CMR beat annotations follow and report it, because a tracker that locks to a different level would otherwise be counted as wrong.

**Creating the Tamil film music test set.** We will select 30 clips (40 if time allows) from different decades and with different rhythmic styles. For each song we take one 30 second clip. The beats are annotated by tapping along while the clip plays, and each tap is then corrected by looking at the waveform in Sonic Visualiser. The annotation guideline is as follows. Beats are tapped at the main pulse a listener would clap to, which is usually set by the kick and snare drums. When two levels feel equally natural, the slower one is chosen and the clip is flagged. For each clip we record the song, film, year, composer, audio source (commercial release or a named online upload), start time and duration. No audio is stored in the repository. To check reliability, 10 clips are annotated a second time on a different day, and a second group member annotates the same 10 clips independently. The annotation files use the same format as the GTZAN annotations, so that all evaluation code works on both.

## 6. Methodology

### 6.1 Shared code and ownership

The shared code is small, and each part has one owner. The time for it is counted in each member's workload (Section 12).

- **Irai:** loaders for GTZAN tempo, GiantSteps and the Tamil film set, and the Accuracy1 and Accuracy2 functions, which mir_eval does not provide.
- **Nalluraj:** loaders for CMR and Saraga, the mir_eval wrapper that writes results, and the click track utility.
- **Usha:** the GTZAN beat loader and the GTZAN train and test split, committed by Oct 25.

All results are saved in one CSV format with the columns track, dataset, genre or tala, method, metric and value. No performance indicator depends on code written by another member. Each method takes its inputs either from its own owner or from librosa and madmom.

### 6.2 Onset strength envelope

The autocorrelation and dynamic programming methods start from an onset strength envelope. This is a signal that rises sharply when a new note or drum hit starts. It is computed from the change in a log magnitude spectrogram over time. The envelope turns the beat problem into a simpler one: find a regular pattern of peaks. The neural network does not use it. It learns its own features from spectrogram frames.

### 6.3 Tempo estimation by autocorrelation (Irai)

Autocorrelation compares the onset envelope with a shifted copy of itself. If the music repeats every half second, the two copies line up well at a shift of half a second, which corresponds to 120 BPM. The tempo estimate is the shift with the strongest match within a range of 40 to 250 BPM. The main weakness is that a song at 120 BPM also matches well at 60 and 240 BPM. These half and double tempo mistakes are called octave errors. Percival and Tzanetakis [4] reduce them with two extra stages. The first cross-correlates the envelope with ideal pulse trains at each candidate tempo. The second makes a final octave decision. We implement the pulse train stage as an expected indicator and the octave decision as an advanced one.

### 6.4 Beat tracking by dynamic programming (Nalluraj)

Following Ellis [5], beat tracking is treated as a search problem. The tracker looks for the sequence of beat times that lands on strong onsets while keeping the spacing between beats close to the expected beat period. Dynamic programming solves this efficiently by keeping, for every point in time, the best score of any beat sequence that ends there. A tightness parameter controls how strictly the spacing must follow the tempo. The tracker is run with two tempo inputs: an estimate from librosa, and an oracle tempo taken from the annotations. Comparing the two separates errors in the tempo from errors in placing the beats. The method assumes a steady tempo, which makes it a good test case for Carnatic music, where the tempo can drift.

### 6.5 Neural network beat tracking (Usha)

Instead of hand written rules, a neural network learns what a beat sounds like from annotated examples. We will train a temporal convolutional network (TCN) [6] that takes log mel spectrogram frames as input and outputs the probability of a beat at each frame. Beat times are picked from the peaks of this output. The same output is also passed through madmom's dynamic Bayesian network (DBN), so that the effect of the network and the effect of the post-processing can be compared separately. The network is trained on the GTZAN training split and tested on the held out tracks.

### 6.6 Reference systems

We compare our methods with widely used systems: librosa (tempo and beat tracking), madmom [7] (RNN beat tracker with DBN post-processing, and its joint beat and downbeat tracker [8]) and the 1D State Space tracker [9]. The 1D State Space tracker works online on BeatNet activations, so it is reported separately from the offline trackers. librosa's beat tracker implements the same method as Ellis [5], so Nalluraj uses it to validate his implementation rather than as an independent baseline.

### 6.7 Sama finding

For Carnatic music, finding the sama is similar to finding the downbeat (the first beat of the bar) in Western music. We will run madmom's downbeat tracker first with its default Western setting of 3 or 4 beats per bar, and then with the correct number of beats for each tala and a tempo range set per tala. The downbeat network was trained on Western music, so changing the bar length only changes the bar model, not what the network detects. We will also build a simple sama finder that groups beats into cycles of the known length and chooses the starting beat whose position has the strongest onsets on average.

### 6.8 Listening checks

Throughout the project we will create click tracks, which are the original audio with a short tick added at every predicted beat. Listening to these is the fastest way to catch bugs and to understand failures that numbers alone do not explain.

## 7. Tools

- Python 3, NumPy, SciPy, Matplotlib, pandas, Jupyter
- librosa (https://librosa.org): audio loading, spectrograms, onset strength, reference tempo and beat tracking
- madmom: reference beat, downbeat and tempo trackers
- 1D State Space tracker: https://github.com/mjhydri/1D-StateSpace
- PyTorch: training the TCN beat tracker, with Google Colab for GPU access if needed
- mir_eval [10]: evaluation metrics
- mirdata: dataset loaders
- Sonic Visualiser (https://www.sonicvisualiser.org): beat annotation for the Tamil film music test set
- Marsyas (https://github.com/marsyas/marsyas): reference implementation of the tempo estimator in [4]
- GitHub: version control, issue tracking and collaboration

### 7.1 Reproducibility

All experiments use one Python version and the packages listed in `requirements.txt`. Exact versions will be pinned in a lock file with the first committed code. madmom and the 1D State Space tracker may need their own environment, which will be documented if so. Random seeds are fixed at 0. GTZAN results use the committed split file. The dataset versions used are recorded: the gtzan_tempo_beat commit, GiantSteps Tempo annotation version 2, and CMR version 1.0 (full set or 118 excerpt subset). For beat annotated data, the reference tempo is 60 divided by the median inter-beat interval. Timing is measured on CPU on one named machine. Each member will provide one script that rebuilds their results table from the raw data. No audio is stored in the repository.

## 8. Evaluation Metrics

- **Tempo**: Accuracy1 (estimate within 4% of the annotated tempo) and Accuracy2 (the same, but also accepting double, triple, half and one third of the annotated tempo). The gap between the two shows how many errors are octave errors. Both are implemented in our evaluation module.
- **Beat tracking**: F-measure with a 70 ms tolerance window, CMLt and AMLt (which check whether the tracker stays correct over continuous stretches) and Information Gain, computed with mir_eval [10]. mir_eval's default of ignoring beats in the first 5 seconds is kept for all datasets.
- **Sama tracking**: F-measure on the predicted sama positions with a 70 ms window.
- **Annotation reliability**: the same beat metrics, computed between two annotations of the same clip.
- **Efficiency**: processing time per minute of audio, measured on the same machine.

All results will be reported per dataset, per genre for GTZAN, per style for Tamil film music, and per tala for Carnatic music.

## 9. Timeline

| Period | Milestone |
|---|---|
| Oct 19 to Oct 25 | Design specification. Repository setup. CMR access request sent. GiantSteps download run. GTZAN split committed. Shared loaders and evaluation code. |
| Oct 26 to Nov 8 | Basic performance indicators for all members. Tamil film clips selected and guideline written. |
| Nov 9 to Nov 22 | Expected performance indicators in progress. Tamil film annotation complete. First combined results table. |
| Progress report (date TBC) | README updated with first results |
| Nov 23 to Nov 29 | Remaining expected indicators and advanced indicators where time allows |
| Nov 30 to final deadline | Combined results table, hypothesis tests, error analysis, ISMIR report and presentation, with one section written by each member |

The dates on each performance indicator below follow this timeline. Basic and expected indicators form the core plan. Advanced indicators are stretch goals that we will pursue once the core work is complete.

## 10. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| CMR access is delayed | Send the request on Zenodo in the first week and ask the instructor whether a local copy is available. If access is not granted by Nov 8, the Carnatic beat evaluation switches to Saraga at the cycle level: for each pair of consecutive sama marks, the predicted beats are counted and the cycle is scored as correct if the count matches the tala (with separate counts for double and half). Approximate beat references, made by spacing the known number of beats evenly between sama marks, are used only where the cycle length is stable, are checked against 5 hand tapped excerpts, and are reported separately as approximate. |
| GiantSteps audio links no longer work | Run the download script by Oct 25. If fewer than 80% of the previews download, drop GiantSteps and report tempo results on GTZAN and the South Indian data only. |
| madmom fails to install on recent Python versions | Install from the GitHub source, or use a separate environment for madmom and the 1D State Space tracker. |
| Annotating the Tamil film clips takes longer than planned | Commit to 30 clips and treat 40 as a target, while keeping the 10 clip reliability check. |
| TCN training is slow on a laptop | Train on Google Colab. GTZAN is small enough for a free GPU. |

## 11. Proposed Extension and Questions for the Instructor

### 11.1 What we propose

The core of our project (Objective 1 for each member) follows DP2 and uses Western music only. As an extension (Objective 2 for each member), we propose testing the same methods on two kinds of South Indian music:

- **Carnatic music**, using existing datasets with beat and sama annotations.
- **Tamil film music**, using a small test set that we annotate ourselves, since no public annotated dataset seems to exist.

This would let us study how methods built for Western music behave on music with different rhythmic structure. We think this makes the project more useful and more original than running the standard DP2 pipeline alone. Since it goes beyond the provided resources, we would like to confirm with you that it is feasible before we commit. The schedule for the extension is part of the timeline in Section 9. Annotating the Tamil film clips is expected to take about 30 minutes per clip including correction and export, so about 15 to 20 hours in total.

### 11.2 How you could help

- Point us to Carnatic or other Indian music datasets with beat or tempo annotations, or share a local copy of the CMR dataset if you have one.
- Suggest literature on rhythm analysis of Indian music, or of popular music outside the Western tradition.
- Give feedback on our annotation guideline for the Tamil film clips (Section 5).
- Point us to the Marsyas implementation of your tempo estimator [4], so that Irai can check his implementation against it.

### 11.3 Questions on feasibility

1. **Time:** is the extension realistic within the term alongside the core DP2 work? Would you prefer that we reduce it, for example to Carnatic music only or Tamil film music only?
2. **Datasets:** is the CompMusic Carnatic Rhythm dataset a good choice, and is it acceptable to use it for a course project? Do you have a local copy we could use, as with GTZAN? Is there a better dataset you would recommend?
3. **Annotation:** is a self-annotated test set of 30 to 40 clips, with a reliability check, acceptable as evaluation data? We plan to publish only the annotations and the song list, not the audio. Is that appropriate?
4. **Scope and grading:** would this extension be seen as going beyond a default project? Should we list it as a student proposal instead of DP2?
5. **Workload:** our estimate is about 76, 71 and 76 hours per member (Section 12). Does this look balanced and independent enough to you?
6. **Deadlines:** the assignment mentions a progress report at the end of February. Could you confirm the progress report and final deadlines for this term, so we can fix our dates?

### 11.4 Fallback plan

If the extension is not feasible, each member's Objective 2 would be replaced with a Western-only objective of similar size:

- **Irai:** tempo estimation under known tempo changes, using GTZAN tracks time-stretched by fixed factors, measuring how accurately and how quickly each estimator follows the new tempo.
- **Nalluraj:** robustness of beat tracking to degraded audio (added noise, low pass filtering and lossy compression).
- **Usha:** downbeat tracking on GTZAN, comparing madmom with a simple downbeat finder she builds.

---

## 12. Individual Objectives

Each member has one objective focused on building a method from scratch (core DP2) and one objective focused on a part of the South Indian evaluation (proposed extension, see Section 11). The objectives are independent of each other. Our estimate of the workload, including the shared code in Section 6.1, is:

| Member | Objective 1 | Objective 2 | Shared code and integration | Total |
|---|---|---|---|---|
| Irai | 36 h | 34 h | 6 h | about 76 h |
| Nalluraj | 30 h | 30 h | 11 h | about 71 h |
| Usha | 43 h | 25 h | 8 h | about 76 h |

These are estimates and may change by about 20%. Nalluraj's shared work includes the second annotation of the 10 Tamil film reliability clips.

## Irai Kumaran Sivanesan

Role: tempo estimation and the Tamil film music test set

* Objective 1 (implement an autocorrelation based tempo estimator from scratch and evaluate it on GTZAN and GiantSteps)
   * PI1 (basic): compute an onset strength envelope and its autocorrelation, and select the strongest peak between 40 and 250 BPM as the tempo estimate (Nov 1)
   * PI2 (basic): implement Accuracy1 and Accuracy2, report both on the GTZAN test split and on GiantSteps, and count the half and double tempo errors (Nov 8)
   * PI3 (expected): add the pulse train cross-correlation stage described by Percival and Tzanetakis [4] and measure the change in Accuracy1 and in the number of octave errors on the GTZAN test split (Nov 15)
   * PI4 (expected): compare against librosa and madmom tempo estimates on the same tracks, with a per-genre results table for GTZAN and CPU timing for each method (Nov 22)
   * PI5 (advanced): add the octave decision stage described by Percival and Tzanetakis [4] and measure how many of the remaining half and double tempo errors it removes (Nov 29)

* Objective 2 (create and evaluate a beat annotated Tamil film music test set)
   * PI1 (basic): select 30 clips of 30 seconds from Tamil film songs across several decades (40 if time allows), write the annotation guideline, and record the song list with source and start time for each clip (Nov 1)
   * PI2 (basic): annotate the beats of every clip by tapping and correcting the taps in Sonic Visualiser, and export them in the same format as the GTZAN annotations (Nov 15)
   * PI3 (expected): annotate 10 clips a second time on a different day and score the two versions against each other with mir_eval, aiming for an F-measure of at least 0.9, and also report agreement with a second group member's independent annotation of the same clips (Nov 22)
   * PI4 (expected): derive a reference tempo for each clip as 60 divided by the median inter-beat interval of the annotations, run the from-scratch tempo estimator, the librosa and madmom tempo estimators, and the librosa and madmom beat trackers on all clips, and report Accuracy1, Accuracy2, F-measure, CMLt and AMLt next to the GTZAN results (Nov 29)
   * PI5 (advanced): label each clip by rhythm style (Western kit, Indian percussion or mixed) using the definitions in the guideline, and report accuracy per style with bootstrap confidence intervals (Nov 29)

## Nalluraj Babu

Role: dynamic programming beat tracking and Carnatic beat evaluation

* Objective 1 (implement a dynamic programming beat tracker from scratch and evaluate it on GTZAN)
   * PI1 (basic): compute the onset strength envelope with librosa and prepare two fixed tempo inputs for the tracker, librosa's global tempo estimate and an oracle tempo taken from the annotations, and check both on 5 GTZAN tracks with click tracks (Nov 1)
   * PI2 (basic): implement the dynamic programming recursion and backtracking described by Ellis [5] to produce beat times (Nov 8)
   * PI3 (expected): evaluate the tracker with mir_eval (F-measure, CMLt and AMLt) on the GTZAN test split, broken down by genre, with both tempo inputs, so that tempo errors are separated from beat placement errors (Nov 15)
   * PI4 (expected): validate the tracker against librosa beat_track, which implements the same algorithm (aiming for an F-measure of at least 0.95 between the two outputs), then compare it with madmom's RNN and DBN beat tracker on the GTZAN test split, and sweep the tightness parameter over at least 6 values with a per-genre F-measure plot (Nov 22)
   * PI5 (advanced): extend the tracker so it can follow a slowly changing tempo, and test it on GTZAN tracks with tempo variation and on GTZAN test tracks that are gradually time-stretched by a known amount (Nov 29)

* Objective 2 (evaluate beat tracking on Carnatic music)
   * PI1 (basic): load the Carnatic excerpts and their beat annotations, check them with click tracks, and document which beat level (kalai) the annotations follow (Nov 8, or one week after CMR access)
   * PI2 (basic): run the dynamic programming tracker, librosa and the madmom beat tracker on all excerpts and report F-measure, CMLt and AMLt for each tala (Nov 15)
   * PI3 (expected): for every excerpt and every tracker, classify the metrical level the tracker locks to (annotated beat, double, half, triple, third or other) from the ratio of the median predicted to the median annotated inter-beat interval with an 8% tolerance, and report the distribution for each tala (Nov 22)
   * PI4 (expected): compare onset envelopes of Carnatic and GTZAN tracks, and test whether an envelope computed from the percussive part after harmonic-percussive separation improves tracking on Carnatic music (Nov 29)
   * PI5 (advanced): measure the tempo drift across the cycles of each excerpt, test whether the trackers fail more often on excerpts with more drift, and run the tempo-following tracker from Objective 1 on the excerpts with the most drift (Nov 29)

## Usha Sundaresan Manivannan

Role: neural network beat tracking and Carnatic sama tracking

* Objective 1 (train a neural network beat tracker from scratch and compare it with established systems)
   * PI1 (basic): build a training pipeline that turns GTZAN audio into log mel spectrogram frames with beat labels, using the artist-filtered split of Kereliuk et al. [3] committed to the repository (Nov 1)
   * PI2 (basic): implement and train a small temporal convolutional network (TCN) beat tracker in PyTorch, following Davies and Böck [6] (Nov 8)
   * PI3 (expected): turn the network output into beat times with peak picking and evaluate it on the GTZAN test split with mir_eval (Nov 15)
   * PI4 (expected): compare the network with madmom's RNN and DBN beat tracker and the 1D State Space tracker in accuracy and CPU processing time per minute of audio, and also pass the network's output through madmom's DBN to separate the effect of the network from that of the post-processing (Nov 22)
   * PI5 (advanced): add data augmentation by time stretching and test whether it improves accuracy on tracks with tempos that are rare in the training data (Nov 29)

* Objective 2 (find the sama of the tala cycle in Carnatic music)
   * PI1 (basic): load the sama annotations and run madmom's downbeat tracker with its default bar lengths of 3 and 4 beats (Nov 8, or one week after CMR access)
   * PI2 (basic): evaluate the predicted downbeats against the annotated sama positions with F-measure (70 ms window) for each tala, and report how often the tracker places the downbeat on a beat other than the sama (Nov 15)
   * PI3 (expected): rerun the downbeat tracker with the correct number of beats per cycle for each tala (8, 3, 7 or 5) and a tempo range set per tala, both on its own beat estimates and on the annotated beats, so that beat tracking errors are separated from errors in finding the start of the cycle (Nov 22)
   * PI4 (expected): implement a simple sama finder that groups madmom's beat estimates into cycles of the known length and picks the starting beat whose position has the strongest average onset strength, then compare it with the madmom downbeat tracker (Nov 29)
   * PI5 (advanced): run the GTZAN-trained TCN from Objective 1 on the Carnatic excerpts, report F-measure, CMLt and AMLt per tala next to madmom, and test whether using the TCN's beat activations improves the sama finder (Nov 29)

---

## 13. Related Work

Gouyon et al. [11] ran the first large comparison of tempo induction algorithms. That study set Accuracy1 and Accuracy2 as standard metrics and showed that choosing the wrong metrical level (half or double tempo) is the most common error. Most classic systems first compute an onset detection function and then look for repeating patterns in it. Percival and Tzanetakis [4] proposed a streamlined estimator based on autocorrelation and cross-correlation with pulse trains, which performed close to much more complex systems. Ellis [5] treated beat tracking as dynamic programming. The method searches for the beat sequence that best matches the onsets while keeping a steady spacing between beats.

Machine learning has since become the dominant approach. Böck and Schedl [12] used recurrent neural networks to predict where beats occur. Böck et al. [8] extended this to joint beat and downbeat tracking with a dynamic Bayesian network. Davies and Böck [6] showed that temporal convolutional networks track beats accurately with fewer parameters. Heydari et al. [9] proposed a one dimensional state space that makes inference much faster and can run online. Davies et al. [13] give a practical overview of the field in their ISMIR tutorial. It also redistributes the GTZAN beat annotations of Marchand and Peeters [1], which we use.

Only a small part of this work looks beyond Western music. Holzapfel et al. [14] showed that a meter tracking model can handle the "odd" meters found in Turkish, Indian and Greek music when it is trained on that music. Srinivasamurthy et al. [15] compared rhythm analysis methods on Turkish and Indian art music and found that methods built for Western music struggle with long and irregular cycles. Srinivasamurthy and Serra [16] proposed a supervised method for tracking tala cycles in Carnatic music, using the full CMR dataset, and Srinivasamurthy et al. [17] made this kind of tracking much faster with particle filters, using the 118 excerpt subset. These studies use specialised Bayesian models on classical music. Our project differs in two ways. We test the general purpose tools that most people use today on the same Carnatic data and compare them directly with their behaviour on Western data. We also add Tamil film music, a popular style that mixes both traditions and has no public beat annotations.

For evaluation, Raffel et al. [10] provide reference implementations of the standard beat tracking metrics in mir_eval. Our Western datasets are GTZAN, introduced by Tzanetakis and Cook [18], and GiantSteps Tempo by Knees et al. [19]. Sturm [2] documents known problems with GTZAN, and Kereliuk et al. [3] proposed the artist-filtered split we use to avoid them. Our Carnatic data comes from the CompMusic project [16, 17] and the Saraga collection [20]. Our main reference software is madmom [7].

## 14. References

[1] U. Marchand, Q. Fresnel, and G. Peeters, "GTZAN-Rhythm: Extending the GTZAN test-set with beat, downbeat and swing annotations," in Late-Breaking Demo Session of ISMIR, 2015.

[2] B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use," arXiv preprint arXiv:1306.1461, 2013.

[3] C. Kereliuk, B. L. Sturm, and J. Larsen, "Deep learning and music adversaries," IEEE Transactions on Multimedia, vol. 17, no. 11, pp. 2059-2071, 2015.

[4] G. Percival and G. Tzanetakis, "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 1765-1776, 2014.

[5] D. P. W. Ellis, "Beat tracking by dynamic programming," Journal of New Music Research, vol. 36, no. 1, pp. 51-60, 2007.

[6] M. E. P. Davies and S. Böck, "Temporal convolutional networks for musical audio beat tracking," in Proc. European Signal Processing Conference (EUSIPCO), 2019.

[7] S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer, "madmom: A new Python audio and music signal processing library," in Proc. ACM International Conference on Multimedia, pp. 1174-1178, 2016.

[8] S. Böck, F. Krebs, and G. Widmer, "Joint beat and downbeat tracking with recurrent neural networks," in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2016.

[9] M. Heydari, M. McCallum, A. Ehmann, and Z. Duan, "A novel 1D state space for efficient music rhythmic analysis," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[10] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, "mir_eval: A transparent implementation of common MIR metrics," in Proc. ISMIR, 2014.

[11] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, "An experimental comparison of audio tempo induction algorithms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832-1844, 2006.

[12] S. Böck and M. Schedl, "Enhanced beat tracking with context-aware neural networks," in Proc. International Conference on Digital Audio Effects (DAFx), 2011.

[13] M. E. P. Davies, S. Böck, and M. Fuentes, "Tempo, beat and downbeat estimation," Tutorial at ISMIR, 2021. [Online]. Available: https://tempobeatdownbeat.github.io/tutorial

[14] A. Holzapfel, F. Krebs, and A. Srinivasamurthy, "Tracking the 'odd': Meter inference in a culturally diverse music corpus," in Proc. ISMIR, 2014.

[15] A. Srinivasamurthy, A. Holzapfel, and X. Serra, "In search of automatic rhythm analysis methods for Turkish and Indian art music," Journal of New Music Research, vol. 43, no. 1, pp. 94-114, 2014.

[16] A. Srinivasamurthy and X. Serra, "A supervised approach to hierarchical metrical cycle tracking from audio music recordings," in Proc. IEEE ICASSP, 2014.

[17] A. Srinivasamurthy, A. Holzapfel, A. T. Cemgil, and X. Serra, "Particle filters for efficient meter tracking with dynamic Bayesian networks," in Proc. ISMIR, pp. 197-203, 2015.

[18] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, 2002.

[19] P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. Le Goff, "Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections," in Proc. ISMIR, 2015.

[20] A. Srinivasamurthy, S. Gulati, R. Caro Repetto, and X. Serra, "Saraga: Open datasets for research on Indian art music," Empirical Musicology Review, vol. 16, no. 1, 2021.
