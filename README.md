# Beat Tracking and Tempo Estimation (DP2_B)

CSC 475/575 Music Information Retrieval, University of Victoria, Fall 2026

Instructor: George Tzanetakis

Group members:

- Irai Kumaran Sivanesan (iraikumarans@uvic.ca)
- Nalluraj Babu (nallurajbabu@uvic.ca)
- Usha Sundaresan Manivannan (mvannan@uvic.ca)

Repository: https://github.com/Iraikumaran/beat-tracking

---

## 1. Project Overview

Tempo estimation and beat tracking aim to find the rate and the positions of the pulse that a listener would tap along to. They are basic tasks in music information retrieval and a first step for many other analyses. This project is Default Project 2 (Beat Tracking and Tempo Estimation) and covers two related tasks:

- **Tempo estimation**: given an audio clip, output one global tempo value in beats per minute (BPM).
- **Beat tracking**: given an audio clip, output a list of time stamps (in seconds) where the beats occur.

Each group member builds one method from scratch: an autocorrelation tempo estimator, a dynamic programming beat tracker and a neural network beat tracker. We compare these with established systems on the datasets provided for DP2. We then study the problem from three further angles: how tempo estimation copes with tempo changes, how beat tracking copes with degraded audio, and how well the first beat of each bar (the downbeat) can be found.

## 2. Problem Statement

Tempo estimation and beat tracking have been studied for more than twenty years, but several problems are still open:

- **Octave errors.** A song at 120 BPM also has strong periodicities at 60 and 240 BPM. Tempo estimators often pick one of these instead of the tempo a listener would tap. This is the most common error reported in the literature [1].
- **Changing tempo.** Many methods assume that the tempo stays constant. Real performances speed up and slow down, and some songs change tempo suddenly.
- **Weak or unclear beats.** Genres without strong drums, such as classical and jazz, are much harder than rock or dance music, because the onsets that mark the beat are weaker.
- **Real recording conditions.** Audio in everyday use is often noisy, band limited or heavily compressed. Most evaluations use clean audio.
- **The bar level.** Knowing where the beats are does not tell us which beat is the first of the bar. Downbeat tracking is a harder problem that builds on beat tracking.

Our central question is: **how do classic signal processing methods and neural network methods compare on these problems, and where and why does each one fail?**

## 3. Related Work

Gouyon et al. [1] ran the first large comparison of tempo induction algorithms. That study set Accuracy1 and Accuracy2 as standard metrics and showed that choosing the wrong metrical level (half or double tempo) is the most common error. Most classic systems first compute an onset detection function and then look for repeating patterns in it. Percival and Tzanetakis [2] proposed a streamlined estimator based on autocorrelation and cross-correlation with pulse trains, which performed close to much more complex systems. Ellis [3] treated beat tracking as dynamic programming. The method searches for the beat sequence that best matches the onsets while keeping a steady spacing between beats. For music whose tempo changes over time, Grosche and Müller [4] estimate a local pulse in short windows rather than one global tempo.

Machine learning has since become the dominant approach. Böck and Schedl [5] used recurrent neural networks to predict where beats occur. Böck et al. [6] extended this to joint beat and downbeat tracking with a dynamic Bayesian network, and Krebs et al. [7] tracked downbeats from features computed between known beats. Davies and Böck [8] showed that temporal convolutional networks track beats accurately with fewer parameters. Heydari et al. [9] proposed a one dimensional state space that makes inference much faster and can run online. Davies et al. [10] give a practical overview of the field in their ISMIR tutorial. It also redistributes the GTZAN beat and downbeat annotations of Marchand et al. [11], which we use.

Robustness to real recording conditions is less studied. Mauch and Ewert [12] introduced the Audio Degradation Toolbox and showed that several MIR systems, including beat trackers, lose accuracy under common degradations. Our robustness study follows their approach on a smaller scale and adds changes to the tracker that aim to recover the lost accuracy.

For evaluation, Davies et al. [13] define the beat tracking metrics we use, and Raffel et al. [14] provide their reference implementation in mir_eval. Our datasets are GTZAN, introduced by Tzanetakis and Cook [15], and GiantSteps Tempo by Knees et al. [16]. Sturm [17] documents known problems with GTZAN, and Kereliuk et al. [18] proposed the artist-filtered split we use to avoid them. Our main reference software is madmom [19]. Almost all of this work uses Western music. Open annotated collections of other traditions exist, such as the Saraga collection of Carnatic music [20], which motivates the optional question in Section 14.2.

## 4. Motivation

Beat and tempo information is used in many real applications, including DJ software, playlist generation, music games, automatic accompaniment and fitness apps that match music to running pace. It is also a first step for other tasks such as downbeat tracking, chord recognition and structure analysis. Errors at this stage carry through to everything built on top of it.

The project also has a learning goal. Each member will implement one complete method from scratch, so that we understand how these systems work rather than only calling library functions. Comparing a hand built method with established systems shows clearly what each design choice contributes.

## 5. Project Vision and Expected Outcomes

At the end of the term we expect to deliver the following:

1. **Three working methods built from scratch**: an autocorrelation tempo estimator, a dynamic programming beat tracker and a neural network beat tracker trained by us.
2. **A benchmark**: accuracy and speed of our methods compared with librosa, madmom and the 1D State Space tracker on GTZAN and GiantSteps, broken down by genre.
3. **A tempo change study**: how well tempo estimators follow known tempo changes, and a local tempo estimator that tracks tempo over time.
4. **A robustness study**: how beat tracking accuracy drops under noise, low pass filtering and lossy compression, and how much of the loss two changes to the onset envelope can recover.
5. **A downbeat study**: how well the first beat of each bar can be found, and whether errors come from the beats or from the bar level.
6. **A final report** in ISMIR paper format, and a public repository with code, results and instructions to reproduce them.

Our hypotheses, which we will test, are:

- **H1:** Most tempo errors on GTZAN and GiantSteps are octave errors, and the pulse train stage of Percival and Tzanetakis [2] removes a large part of them.
- **H2:** Neural network beat trackers will beat dynamic programming beat tracking (librosa's implementation of Ellis [3]) by at least 0.10 F-measure on genres with weak percussion (classical, jazz), while the gap will be smaller than 0.05 on rock, disco and pop.
- **H3:** Beat tracking accuracy will drop more under added noise than under low pass filtering, because the kick and bass that carry the beat in most GTZAN genres lie below the filter cutoffs.

## 6. Datasets

| Dataset | Content | Annotations | Access | Used for |
|---|---|---|---|---|
| GTZAN | 1000 clips of 30 s in 10 genres | Beats, downbeats and tempo [11] | Audio from the instructor, annotations from the gtzan_tempo_beat repository | All members |
| GiantSteps Tempo | 664 electronic dance music previews of about 2 minutes | Tempo, version 2 annotations | Audio fetched with the repository's download script from Beatport preview links | Tempo estimation |

Links:

- GTZAN annotations: https://github.com/TempoBeatDownbeat/gtzan_tempo_beat
- GiantSteps Tempo: https://github.com/GiantSteps/giantsteps-tempo-dataset

**GTZAN split.** GTZAN contains repeated songs and several clips from the same artists [17]. To avoid results that are too optimistic, all GTZAN results are reported on the artist-filtered, fault-filtered train, validation and test split of Kereliuk et al. [18], which uses a cleaned subset of the 1000 clips. The split will be committed as `data/splits/gtzan_split.csv` by Oct 25. The same loader also lists the 30 test tracks whose annotated inter-beat intervals vary the most, in `data/splits/tempo_variation.csv`, so that every member uses the same set of tracks with tempo variation.

**Downbeats.** The beat files in gtzan_tempo_beat have a second column with the position of each beat in its bar, where 1 marks the downbeat. The repository covers 999 tracks (reggae.00086 is missing). Six jazz files have beats but no bar positions. Bars have 4 beats in 933 files, 3 in 49, 2 in 7 and 5 in 4, and 15 files change bar length within the clip.

**Derived data.** Two of our studies create new versions of GTZAN test tracks with known properties, so that the correct answer is known exactly:

- **Time-stretched tracks** (Irai): tracks stretched by fixed factors, and tracks made by joining two stretched segments at an annotated beat, so that the tempo and the time of the tempo change are known.
- **Degraded tracks** (Nalluraj): tracks with added noise, low pass filtering and lossy compression at fixed levels, created in the style of the Audio Degradation Toolbox [12].

No audio, original or derived, is stored in the repository. Only the scripts that create the derived data are committed, in `scripts/`.

## 7. Methodology

### 7.1 Shared code and ownership

The shared code is small, and each part has one owner. The time for it is counted in each member's workload (Section 12).

- **Irai:** loaders for GTZAN tempo and GiantSteps, and the Accuracy1 and Accuracy2 functions, which mir_eval does not provide.
- **Nalluraj:** a small mir_eval wrapper that writes results in the shared CSV format, and the click track utility.
- **Usha:** the GTZAN beat and downbeat loader, the GTZAN train, validation and test split, and the list of tracks with tempo variation, committed by Oct 25.

All results are saved in one CSV format with the columns track, dataset, genre, method, condition, metric and value. Members may also call mir_eval directly, as long as they use the same format. No performance indicator depends on a method written by another member. Each method takes its inputs either from its own owner or from librosa and madmom.

### 7.2 Onset strength envelope

The autocorrelation and dynamic programming methods start from an onset strength envelope. This is a signal that rises sharply when a new note or drum hit starts. It is computed from the change in a log magnitude spectrogram over time. The envelope turns the beat problem into a simpler one: find a regular pattern of peaks. The neural network does not use it. It learns its own features from spectrogram frames.

### 7.3 Tempo estimation by autocorrelation (Irai)

Autocorrelation compares the onset envelope with a shifted copy of itself. If the music repeats every half second, the two copies line up well at a shift of half a second, which corresponds to 120 BPM. The tempo estimate is the shift with the strongest match within a range of 40 to 250 BPM. The main weakness is the octave error described in Section 2. Percival and Tzanetakis [2] reduce it with two extra stages. The first cross-correlates the envelope with ideal pulse trains at each candidate tempo. The second makes a final octave decision. We implement the pulse train stage as an expected indicator and the octave decision as an advanced one. For music with changing tempo, the same idea is applied in short sliding windows to give a local tempo curve, in the spirit of [4].

### 7.4 Beat tracking by dynamic programming (Nalluraj)

Following Ellis [3], beat tracking is treated as a search problem. The tracker looks for the sequence of beat times that lands on strong onsets while keeping the spacing between beats close to the expected beat period. Dynamic programming solves this efficiently by keeping, for every point in time, the best score of any beat sequence that ends there. A tightness parameter controls how strictly the spacing must follow the tempo. The tracker is run with two tempo inputs: an estimate from librosa, and an oracle tempo taken from the annotations. Comparing the two separates errors in the tempo from errors in placing the beats.

### 7.5 Neural network beat tracking (Usha)

Instead of hand written rules, a neural network learns what a beat sounds like from annotated examples. We will train a temporal convolutional network (TCN) [8] that takes log mel spectrogram frames as input and outputs the probability of a beat at each frame. Beat times are picked from the peaks of this output. The same output is also passed through madmom's dynamic Bayesian network (DBN), so that the effect of the network and the effect of the post-processing can be compared separately. The network is trained on the GTZAN training split and tested on the held out tracks.

### 7.6 Reference systems

We compare our methods with widely used systems: librosa (tempo and beat tracking), madmom [19] (RNN beat tracker [5] with DBN post-processing, and its downbeat trackers [6, 7]) and the 1D State Space tracker [9]. The 1D State Space tracker works online on BeatNet activations, so it is reported separately from the offline trackers. librosa's beat tracker implements the same method as Ellis [3], so it serves as the dynamic programming reference and as a check on Nalluraj's implementation, not as an independent method. We will confirm from the madmom documentation and [6, 7] which datasets its models were trained on, and report any overlap with GTZAN.

### 7.7 Downbeat tracking (Usha)

A downbeat is the first beat of a bar. We will run madmom's joint beat and downbeat tracker on GTZAN and evaluate it against the GTZAN-Rhythm downbeat annotations [11]. To separate beat errors from bar errors, madmom's bar tracker [7] is also run on the annotated beats. We will also build a simple downbeat finder that groups beats into bars of the annotated length and chooses the starting beat whose position has the strongest low frequency onsets on average, since the kick drum often falls on the downbeat in popular music.

### 7.8 Listening checks

Throughout the project we will create click tracks, which are the original audio with a short tick added at every predicted beat. Listening to these is the fastest way to catch bugs and to understand failures that numbers alone do not explain.

## 8. Tools

- Python 3, NumPy, SciPy, Matplotlib, pandas, Jupyter
- librosa (https://librosa.org): audio loading, spectrograms, onset strength, harmonic-percussive separation, reference tempo and beat tracking
- madmom: reference beat, downbeat and tempo trackers
- 1D State Space tracker (https://github.com/mjhydri/1D-StateSpace), which needs the pretrained BeatNet models and PyAudio
- PyTorch: training the TCN beat tracker, with Google Colab for GPU access if needed
- Rubber Band through pyrubberband, or librosa's phase vocoder: time stretching, checked with a stretch factor of 1.0 as a control
- ffmpeg with the LAME MP3 encoder: lossy compression at fixed bit rates
- mir_eval [14]: evaluation metrics
- mirdata: dataset loaders
- Marsyas (https://github.com/marsyas/marsyas): reference implementation of the tempo estimator in [2]
- GitHub: version control, issue tracking and collaboration

### 8.1 Reproducibility

All experiments use one Python version and the packages listed in `requirements.txt`. Exact versions, including the ffmpeg and Rubber Band versions, will be pinned in a lock file with the first committed code. madmom and the 1D State Space tracker may need their own environment, which will be documented if so. Random seeds are fixed at 0. GTZAN results use the committed split file. The dataset versions used are recorded: the gtzan_tempo_beat commit and GiantSteps Tempo annotation version 2. Derived data is created by committed scripts with fixed parameters. For beat annotated data, the reference tempo is 60 divided by the median inter-beat interval. Timing is measured on CPU on one named machine. Each member will provide one script that rebuilds their results table from the raw data. No audio is stored in the repository.

## 9. Evaluation Metrics

- **Tempo**: Accuracy1 (estimate within 4% of the annotated tempo) and Accuracy2 (the same, but also accepting double, triple, half and one third of the annotated tempo). The gap between the two shows how many errors are octave errors. Both are implemented in our evaluation module.
- **Local tempo**: mean absolute error in BPM between the estimated and the known tempo curve, the time the estimate takes to settle within 4% after a tempo change, and the error in seconds when the time of the change is detected automatically.
- **Beat tracking**: F-measure with a 70 ms tolerance window, CMLt and AMLt (which check whether the tracker stays correct over continuous stretches) and Information Gain, computed with mir_eval [14] following the definitions of Davies et al. [13]. mir_eval's default of ignoring beats in the first 5 seconds is kept, so that our GTZAN results can be compared with published results.
- **Downbeat tracking**: F-measure on the predicted downbeats with a 70 ms window. The first 5 seconds are not trimmed for downbeats, because a 30 second clip contains only about 15 bars.
- **Efficiency**: processing time per minute of audio, measured on the same machine.

All results will be reported per dataset and per genre for GTZAN.

## 10. Timeline

| Period | Milestone |
|---|---|
| Oct 19 to Oct 25 | Design specification. Repository setup. GiantSteps download run. GTZAN split and tempo variation list committed. Shared loaders and evaluation code. |
| Oct 26 to Nov 8 | Basic performance indicators for all members |
| Nov 9 to Nov 22 | Expected performance indicators in progress. First combined results table. |
| Progress report (date to be confirmed, see Section 14.1) | README updated with first results in Section 13 |
| Nov 23 to Nov 29 | Remaining expected indicators and advanced indicators where time allows |
| Nov 30 to final deadline (to be confirmed) | Combined results table, hypothesis tests, error analysis, ISMIR report and presentation, with one section written by each member |

The dates on each performance indicator below follow this timeline. Basic and expected indicators form the core plan. Advanced indicators are stretch goals that we will pursue once the core work is complete. We will update the dates as soon as the progress report and final deadlines are confirmed.

## 11. Risks and Mitigation

| Risk | Mitigation |
|---|---|
| GiantSteps audio links no longer work | Run the download script by Oct 25. If fewer than 80% of the previews download, drop GiantSteps and report tempo results on GTZAN only. |
| GTZAN audio arrives late | Start with GiantSteps and the code that does not need audio (loaders, metrics, split). |
| Some GTZAN tracks have no downbeat annotations or unusual bar lengths | Downbeat results use only the annotated tracks in the test split. Tracks with 2 or 5 beat bars, or with changing bar length, are reported separately. |
| Time stretching adds artifacts that change the onsets | Include a stretch factor of 1.0 as a control and check stretched tracks by ear with click tracks. |
| madmom fails to install on recent Python versions | Install from the GitHub source, or use a separate environment for madmom and the 1D State Space tracker. |
| TCN training is slow on a laptop | Train on Google Colab. GTZAN is small enough for a free GPU. |
| The TCN downbeat output does not train well because downbeat labels are sparse | Weight the downbeat loss, or apply the PI4 downbeat finder to the TCN beats instead. |

---

## 12. Individual Objectives

Each member has two objectives. The first builds one method from scratch and benchmarks it. The second studies one of the open problems from Section 2 and includes a further component that the member builds. The objectives are independent of each other. Our estimate of the workload, including the shared code in Section 7.1, is:

| Member | Objective 1 | Objective 2 | Shared code and integration | Total |
|---|---|---|---|---|
| Irai | 34 h | 31 h | 6 h | about 71 h |
| Nalluraj | 30 h | 30 h | 7 h | about 67 h |
| Usha | 42 h | 27 h | 6 h | about 75 h |

These are estimates and may change by about 20%.

## Irai Kumaran Sivanesan

Role: tempo estimation

* Objective 1 (implement an autocorrelation based tempo estimator from scratch and evaluate it on GTZAN and GiantSteps)
   * PI1 (basic): compute an onset strength envelope and its autocorrelation, and select the strongest peak between 40 and 250 BPM as the tempo estimate (Nov 1)
   * PI2 (basic): implement Accuracy1 and Accuracy2, report both on the GTZAN test split and on GiantSteps, and count the half and double tempo errors (Nov 8)
   * PI3 (expected): add the pulse train cross-correlation stage described by Percival and Tzanetakis [2] and measure the change in Accuracy1 and in the number of octave errors on the GTZAN test split, to test H1 (Nov 15)
   * PI4 (expected): compare against librosa and madmom tempo estimates on the same tracks, with a per-genre results table for GTZAN and CPU timing for each method (Nov 22)
   * PI5 (advanced): add the octave decision stage described by Percival and Tzanetakis [2] and measure how many of the remaining half and double tempo errors it removes (Nov 29)

* Objective 2 (measure how well tempo estimators follow tempo changes, and build a local tempo estimator)
   * PI1 (basic): write a script that time-stretches GTZAN test tracks by 0.8, 0.9, 1.0, 1.1 and 1.2, where 1.0 passes through the same processing as a control, scales the annotated beats by the same factor, and checks on 10 tracks with click tracks that the scaled beats still line up (Nov 1)
   * PI2 (basic): run the from-scratch estimator, librosa and madmom on the stretched tracks, report Accuracy1 and Accuracy2 for each stretch factor, and check whether stretching changes the number of octave errors (Nov 8)
   * PI3 (expected): extend the estimator to a local tempo curve by applying it in sliding windows, and choose the window length that gives the lowest error on the stretched tracks (Nov 22)
   * PI4 (expected): create test tracks with one sudden tempo change by joining two segments stretched by different factors at an annotated beat with a 10 ms crossfade, and measure the local tempo error and settle time of the from-scratch local estimator and of librosa's local tempo estimate (Nov 26)
   * PI5 (advanced): detect the time of the tempo change automatically from the local tempo curve and report the detection error in seconds on the joined tracks, then compare the curve with librosa's tempogram based local tempo on the tracks in tempo_variation.csv (Nov 29)

## Nalluraj Babu

Role: dynamic programming beat tracking and robustness

* Objective 1 (implement a dynamic programming beat tracker from scratch and evaluate it on GTZAN)
   * PI1 (basic): compute the onset strength envelope with librosa and prepare two fixed tempo inputs for the tracker, librosa's global tempo estimate and an oracle tempo taken from the annotations, and check both on 5 GTZAN tracks with click tracks (Nov 1)
   * PI2 (basic): implement the dynamic programming recursion and backtracking described by Ellis [3] to produce beat times (Nov 8)
   * PI3 (expected): evaluate the tracker with mir_eval (F-measure, CMLt and AMLt) on the GTZAN test split, broken down by genre, with both tempo inputs, so that tempo errors are separated from beat placement errors (Nov 15)
   * PI4 (expected): validate the tracker against librosa beat_track, which implements the same algorithm (aiming for an F-measure of at least 0.95 between the two outputs), then compare it with madmom's RNN and DBN beat tracker on the GTZAN test split, and sweep the tightness parameter over at least 6 values with a per-genre F-measure plot (Nov 22)
   * PI5 (advanced): extend the tracker so it can follow a slowly changing tempo, and test it on the tracks in tempo_variation.csv and on GTZAN test tracks with a gradual tempo ramp of known slope, made with his own script (Nov 29)

* Objective 2 (measure how robust beat tracking is to degraded audio, and build changes that recover the lost accuracy)
   * PI1 (basic): write a script that creates degraded versions of the GTZAN test tracks with pink noise at 20, 10, 5 and 0 dB SNR, low pass filtering at 4, 2 and 1 kHz, and MP3 compression at 128, 64 and 32 kbps (Nov 1)
   * PI2 (basic): run the dynamic programming tracker on every degraded version with both tempo inputs from Objective 1, and report F-measure and AMLt for each condition, so that losses caused by a wrong tempo are separated from losses in beat placement (Nov 15)
   * PI3 (expected): run madmom's RNN and DBN tracker and the 1D State Space tracker on the same conditions, plot F-measure against noise level, filter cutoff and bit rate for all trackers, and test H3 for each genre by comparing the drop under low pass filtering with the drop under noise (Nov 22)
   * PI4 (expected): build two changes to the onset envelope of the dynamic programming tracker, a percussive envelope after harmonic-percussive separation and a spectral flux limited to frequencies below the lowest filter cutoff, and measure how much of the lost F-measure each recovers under every condition, broken down by genre (Nov 26)
   * PI5 (advanced): test whether the best tightness value on clean audio is still best under each degradation, and build a rule that sets tightness from the measured peakiness of the onset envelope, then compare it with a fixed tightness (Nov 29)

## Usha Sundaresan Manivannan

Role: neural network beat tracking and downbeat tracking

* Objective 1 (train a neural network beat tracker from scratch and compare it with established systems)
   * PI1 (basic): build a training pipeline that turns GTZAN audio into log mel spectrogram frames with beat labels, using the artist-filtered train, validation and test split of Kereliuk et al. [18] committed to the repository (Nov 1)
   * PI2 (basic): implement and train a small temporal convolutional network (TCN) beat tracker in PyTorch, following Davies and Böck [8] (Nov 8)
   * PI3 (expected): turn the network output into beat times with peak picking and evaluate it on the GTZAN test split with mir_eval, broken down by genre, next to madmom and librosa beat_track as the neural and dynamic programming references, to test H2 (Nov 15)
   * PI4 (expected): compare the network with madmom's RNN and DBN beat tracker and the 1D State Space tracker in accuracy and CPU processing time per minute of audio, and also pass the network's output through madmom's DBN to separate the effect of the network from that of the post-processing (Nov 22)
   * PI5 (advanced): add data augmentation by time stretching and test whether it improves accuracy on tracks with tempos that are rare in the training data (Nov 29)

* Objective 2 (evaluate and improve downbeat tracking on GTZAN)
   * PI1 (basic): load the GTZAN-Rhythm downbeat annotations and run madmom's joint beat and downbeat tracker on the GTZAN test split (Nov 8)
   * PI2 (basic): evaluate the predicted downbeats with F-measure (70 ms window) for each genre, and report how often the tracker places the downbeat on another beat of the bar (Nov 15)
   * PI3 (expected): run madmom's RNNBarProcessor on the audio together with the annotated beats, decode with DBNBarTrackingProcessor (3 and 4 beats per bar), and compare with PI2, so that beat tracking errors are separated from errors in finding the start of the bar (Nov 22)
   * PI4 (expected): implement a simple downbeat finder that groups madmom's beat estimates into bars of the annotated length and picks the starting beat whose position has the strongest average low frequency onset strength, then compare it with the madmom downbeat tracker (Nov 26)
   * PI5 (advanced): add a downbeat output to the TCN from Objective 1, train it on the GTZAN training split, and compare its downbeat F-measure with madmom and the PI4 finder (Nov 29)

---

## 13. Results

This section will be completed for the progress report, with one subsection per member and a combined results table.

### 13.1 Tempo estimation and tempo changes (Irai)

To be completed.

### 13.2 Dynamic programming beat tracking and robustness (Nalluraj)

To be completed.

### 13.3 Neural network beat tracking and downbeat tracking (Usha)

To be completed.

---

## 14. Questions for the Instructor and a Possible Extension

### 14.1 General questions

1. **Deadlines:** the assignment mentions a progress report at the end of February. Could you confirm the progress report and final deadlines for this term, so we can fix our dates?
2. **Workload:** our estimate is about 71, 67 and 75 hours per member (Section 12). Does this look balanced and independent enough to you?
3. **Marsyas:** could you point us to the Marsyas implementation of your tempo estimator [2], so that Irai can check his implementation against it?
4. **Scope:** within DP2, what would make the project excellent rather than adequate in your view, for example an extra dataset, a stronger baseline or a deeper error analysis?

### 14.2 A possible extension: is it feasible?

Everything above is the DP2 project as given, and none of the objectives depend on what follows. Separately, we would like your opinion on an idea we found interesting.

Most beat tracking research uses Western popular music, where the beat is steady and each bar usually has four beats. South Indian music is organised differently. **Carnatic music** (South Indian classical music) follows a tala, a repeating cycle that can have 3, 5, 7 or 8 beats, sometimes in uneven groups such as 3+2+2. The first beat of the cycle is called the sama. **Tamil film music** mixes Western drum kits with Indian percussion and rhythms. We wondered how the methods in this project would behave on this music, and whether a downbeat tracker could find the sama if it is told the length of the cycle.

Annotated Carnatic data exists, for example the CompMusic Carnatic Rhythm dataset (on request through Zenodo, https://zenodo.org/records/1264394) and the Saraga collection [20]. We could not find any beat annotated dataset of Tamil film music, so it would have to be annotated by hand.

If you think it is worthwhile, the extension would replace three advanced indicators: Irai's Objective 2 PI5 (local tempo on Carnatic excerpts), Nalluraj's Objective 1 PI5 (the tempo-following tracker on Carnatic excerpts) and Usha's Objective 2 PI5 (finding the sama with a known cycle length). A small Tamil film test set (about 30 clips) would be added only if you recommend it, and the annotation would be split evenly, with each member annotating 10 clips and another member checking 3 of them. Our questions are:

1. Is this realistic within the term, or would it put the core DP2 work at risk?
2. Do you know a good Carnatic dataset for beat and tempo work, or have a local copy we could use?
3. Would a small self-annotated test set be acceptable as evaluation data?
4. Would this still count as DP2, or would it become a student proposal?

If the answer is no, we will simply complete the DP2 project as described.

## 15. References

[1] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, "An experimental comparison of audio tempo induction algorithms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832-1844, 2006.

[2] G. Percival and G. Tzanetakis, "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 1765-1776, 2014.

[3] D. P. W. Ellis, "Beat tracking by dynamic programming," Journal of New Music Research, vol. 36, no. 1, pp. 51-60, 2007.

[4] P. Grosche and M. Müller, "Extracting predominant local pulse information from music recordings," IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 6, pp. 1688-1701, 2011.

[5] S. Böck and M. Schedl, "Enhanced beat tracking with context-aware neural networks," in Proc. International Conference on Digital Audio Effects (DAFx), 2011.

[6] S. Böck, F. Krebs, and G. Widmer, "Joint beat and downbeat tracking with recurrent neural networks," in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2016.

[7] F. Krebs, S. Böck, M. Dorfer, and G. Widmer, "Downbeat tracking using beat-synchronous features and recurrent neural networks," in Proc. ISMIR, 2016.

[8] M. E. P. Davies and S. Böck, "Temporal convolutional networks for musical audio beat tracking," in Proc. European Signal Processing Conference (EUSIPCO), 2019.

[9] M. Heydari, M. McCallum, A. Ehmann, and Z. Duan, "A novel 1D state space for efficient music rhythmic analysis," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[10] M. E. P. Davies, S. Böck, and M. Fuentes, "Tempo, beat and downbeat estimation," Tutorial at ISMIR, 2021. [Online]. Available: https://tempobeatdownbeat.github.io/tutorial

[11] U. Marchand, Q. Fresnel, and G. Peeters, "GTZAN-Rhythm: Extending the GTZAN test-set with beat, downbeat and swing annotations," in Late-Breaking Demo Session of ISMIR, 2015.

[12] M. Mauch and S. Ewert, "The Audio Degradation Toolbox and its application to robustness evaluation," in Proc. ISMIR, pp. 83-88, 2013.

[13] M. E. P. Davies, N. Degara, and M. D. Plumbley, "Evaluation methods for musical audio beat tracking algorithms," Technical Report C4DM-TR-09-06, Queen Mary University of London, 2009.

[14] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, "mir_eval: A transparent implementation of common MIR metrics," in Proc. ISMIR, 2014.

[15] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, 2002.

[16] P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. Le Goff, "Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections," in Proc. ISMIR, 2015.

[17] B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use," arXiv preprint arXiv:1306.1461, 2013.

[18] C. Kereliuk, B. L. Sturm, and J. Larsen, "Deep learning and music adversaries," IEEE Transactions on Multimedia, vol. 17, pp. 2059-2072, 2015.

[19] S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer, "madmom: A new Python audio and music signal processing library," in Proc. ACM International Conference on Multimedia, pp. 1174-1178, 2016.

[20] A. Srinivasamurthy, S. Gulati, R. Caro Repetto, and X. Serra, "Saraga: Open datasets for research on Indian art music," Empirical Musicology Review, vol. 16, no. 1, 2021.
