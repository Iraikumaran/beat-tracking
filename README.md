# Beat Tracking and Tempo Estimation (DP2_B)

CSC 475/575, University of Victoria, Fall 2026

Irai Kumaran Sivanesan, Nalluraj Babu, Usha Sundaresan Manivannan

Repository: https://github.com/Iraikumaran/beat-tracking

## Project Description

This is Default Project 2. The goal is to estimate the tempo of a song (in BPM) and find the times of its beats, the way a listener would tap along.

Each of us will build one method from scratch: an autocorrelation tempo estimator, a dynamic programming beat tracker, and a small neural network beat tracker. We will compare them with existing tools (librosa, madmom and the 1D State Space tracker) on GTZAN and GiantSteps. Beyond the basic comparison, each of us will look at one known weak spot of beat tracking: songs that change tempo, noisy or low quality audio, and finding the first beat of each bar (the downbeat).

## Datasets

- **GTZAN**: 1000 clips of 30 seconds in 10 genres. We use the beat, downbeat and tempo annotations from the gtzan_tempo_beat repository [1], and the artist-filtered split of Kereliuk et al. [2] to avoid the known problems with GTZAN [3].
- **GiantSteps Tempo** [4]: 664 electronic dance music clips with tempo labels. The audio is downloaded with the script in the dataset repository.

## Tools

Python with librosa, madmom [5], mir_eval [6] and PyTorch. We also use the 1D State Space tracker [7] and Marsyas as reference systems, and ffmpeg for creating compressed audio.

## Evaluation

For tempo we use Accuracy1 and Accuracy2 [8]. For beats and downbeats we use F-measure, CMLt and AMLt from mir_eval [9]. We report results per genre, since some genres are much harder than others.

## Timeline

| Week | Dates | Plan |
|---|---|---|
| 1 | Oct 20 to 26 | Submit this spec, set up the repository, load the datasets, agree on a shared results format |
| 2 | Oct 27 to Nov 2 | Basic version of each method running on GTZAN |
| 3 | Nov 3 to 9 | Evaluation scripts working, first results for each method |
| 4 | Nov 10 to 16 | Comparisons with librosa and madmom |
| 5 | Nov 17 to 23 | Second objective of each member (tempo changes, noisy audio, downbeats) |
| 6 | Nov 24 to 30 | Finish experiments, advanced goals if time allows |
| 7 | Dec 1 to 7 | Write the report and prepare the presentation |

We will adjust these weeks once the progress report and final deadlines are confirmed.

## Related Work

Gouyon et al. [8] compared many early tempo estimation methods and showed that the most common mistake is picking half or double the correct tempo. Percival and Tzanetakis [10] proposed a simple and accurate tempo estimator based on autocorrelation and pulse trains, which Irai's work follows. Ellis [11] described beat tracking with dynamic programming, which is the basis of Nalluraj's tracker and of librosa's beat tracker.

Neural networks are now the most accurate approach. Böck and Schedl [12] used recurrent networks to find beats, and Böck et al. [13] extended this to downbeats. Davies and Böck [14] showed that temporal convolutional networks (TCNs) work well and are small, which is the model Usha will build. Heydari et al. [7] made beat tracking much faster with a simpler state space. Davies et al. [15] give a practical tutorial that covers all of these methods.

For songs with changing tempo, Grosche and Müller [16] estimate the tempo locally in short windows instead of once per song. For noisy audio, Mauch and Ewert [17] showed that many music analysis tools, including beat trackers, get worse when audio is degraded. GTZAN was introduced by Tzanetakis and Cook [18], and Sturm [3] documents its faults.

## Questions for the Instructor

1. Could you confirm the progress report and final deadlines for this term?
2. Could you point us to the Marsyas implementation of your tempo estimator, so Irai can compare against it?
3. **Possible extension.** We are curious how these methods would work on South Indian music. Carnatic music uses rhythm cycles of 3, 5, 7 or 8 beats, and Tamil film music mixes Western drums with Indian percussion. Annotated Carnatic data exists in the CompMusic and Saraga collections [19]. Would it be feasible and worthwhile to test our methods on this music in place of some of our advanced goals? If not, we will stay with the plan above.

## Irai Kumaran Sivanesan

Role: tempo estimation

* Objective 1 (build an autocorrelation tempo estimator)
   * PI1 (basic): estimate tempo from the autocorrelation of the onset envelope
   * PI2 (basic): report Accuracy1 and Accuracy2 on GTZAN and GiantSteps
   * PI3 (expected): add the pulse train step from Percival and Tzanetakis and measure how many half and double tempo errors it fixes
   * PI4 (expected): compare with librosa and madmom per genre, including run time
   * PI5 (advanced): add the final octave decision step from the same paper

* Objective 2 (tempo estimation when the tempo changes)
   * PI1 (basic): create test clips by speeding up and slowing down GTZAN tracks by known amounts
   * PI2 (basic): measure how well each tempo estimator handles these clips
   * PI3 (expected): turn the estimator into a local tempo estimator using short sliding windows
   * PI4 (expected): test it on clips with one sudden tempo change and measure how quickly it follows
   * PI5 (advanced): detect the time of the tempo change automatically

## Nalluraj Babu

Role: beat tracking with dynamic programming

* Objective 1 (build a dynamic programming beat tracker)
   * PI1 (basic): compute the onset envelope and a tempo estimate as input
   * PI2 (basic): implement the dynamic programming beat tracker from Ellis
   * PI3 (expected): evaluate on GTZAN per genre, using both an estimated and the true tempo to see where errors come from
   * PI4 (expected): check it against librosa's version and compare with madmom
   * PI5 (advanced): extend it to follow a slowly changing tempo

* Objective 2 (beat tracking on noisy and low quality audio)
   * PI1 (basic): create degraded versions of GTZAN clips with added noise, low pass filtering and MP3 compression
   * PI2 (basic): measure how much the tracker's accuracy drops under each degradation
   * PI3 (expected): compare the drop with madmom and the 1D State Space tracker
   * PI4 (expected): try a percussive-only onset envelope to make the tracker more robust
   * PI5 (advanced): choose the tracker's tightness setting automatically based on the audio

## Usha Sundaresan Manivannan

Role: neural network beat tracking and downbeats

* Objective 1 (build and train a neural network beat tracker)
   * PI1 (basic): prepare spectrogram inputs and beat labels from the GTZAN training split
   * PI2 (basic): build and train a small TCN beat tracker in PyTorch
   * PI3 (expected): evaluate it on the GTZAN test split per genre
   * PI4 (expected): compare it with madmom and the 1D State Space tracker in accuracy and speed
   * PI5 (advanced): add time stretching as data augmentation and measure the effect

* Objective 2 (downbeat tracking)
   * PI1 (basic): run madmom's downbeat tracker on GTZAN
   * PI2 (basic): evaluate the downbeats per genre against the annotations
   * PI3 (expected): run the bar tracking step on the true beats to separate beat errors from downbeat errors
   * PI4 (expected): build a simple downbeat finder based on bass and kick drum energy and compare it with madmom
   * PI5 (advanced): add a downbeat output to the TCN from Objective 1

## References

[1] U. Marchand, Q. Fresnel, and G. Peeters, "GTZAN-Rhythm: Extending the GTZAN test-set with beat, downbeat and swing annotations," in Late-Breaking Demo Session of ISMIR, 2015.

[2] C. Kereliuk, B. L. Sturm, and J. Larsen, "Deep learning and music adversaries," IEEE Transactions on Multimedia, vol. 17, pp. 2059-2072, 2015.

[3] B. L. Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its future use," arXiv preprint arXiv:1306.1461, 2013.

[4] P. Knees, Á. Faraldo, P. Herrera, R. Vogl, S. Böck, F. Hörschläger, and M. Le Goff, "Two data sets for tempo estimation and key detection in electronic dance music annotated from user corrections," in Proc. ISMIR, 2015.

[5] S. Böck, F. Korzeniowski, J. Schlüter, F. Krebs, and G. Widmer, "madmom: A new Python audio and music signal processing library," in Proc. ACM International Conference on Multimedia, pp. 1174-1178, 2016.

[6] C. Raffel, B. McFee, E. J. Humphrey, J. Salamon, O. Nieto, D. Liang, and D. P. W. Ellis, "mir_eval: A transparent implementation of common MIR metrics," in Proc. ISMIR, 2014.

[7] M. Heydari, M. McCallum, A. Ehmann, and Z. Duan, "A novel 1D state space for efficient music rhythmic analysis," in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2022.

[8] F. Gouyon, A. Klapuri, S. Dixon, M. Alonso, G. Tzanetakis, C. Uhle, and P. Cano, "An experimental comparison of audio tempo induction algorithms," IEEE Transactions on Audio, Speech, and Language Processing, vol. 14, no. 5, pp. 1832-1844, 2006.

[9] M. E. P. Davies, N. Degara, and M. D. Plumbley, "Evaluation methods for musical audio beat tracking algorithms," Technical Report C4DM-TR-09-06, Queen Mary University of London, 2009.

[10] G. Percival and G. Tzanetakis, "Streamlined tempo estimation based on autocorrelation and cross-correlation with pulses," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 22, no. 12, pp. 1765-1776, 2014.

[11] D. P. W. Ellis, "Beat tracking by dynamic programming," Journal of New Music Research, vol. 36, no. 1, pp. 51-60, 2007.

[12] S. Böck and M. Schedl, "Enhanced beat tracking with context-aware neural networks," in Proc. International Conference on Digital Audio Effects (DAFx), 2011.

[13] S. Böck, F. Krebs, and G. Widmer, "Joint beat and downbeat tracking with recurrent neural networks," in Proc. International Society for Music Information Retrieval Conference (ISMIR), 2016.

[14] M. E. P. Davies and S. Böck, "Temporal convolutional networks for musical audio beat tracking," in Proc. European Signal Processing Conference (EUSIPCO), 2019.

[15] M. E. P. Davies, S. Böck, and M. Fuentes, "Tempo, beat and downbeat estimation," Tutorial at ISMIR, 2021. [Online]. Available: https://tempobeatdownbeat.github.io/tutorial

[16] P. Grosche and M. Müller, "Extracting predominant local pulse information from music recordings," IEEE Transactions on Audio, Speech, and Language Processing, vol. 19, no. 6, pp. 1688-1701, 2011.

[17] M. Mauch and S. Ewert, "The Audio Degradation Toolbox and its application to robustness evaluation," in Proc. ISMIR, pp. 83-88, 2013.

[18] G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, 2002.

[19] A. Srinivasamurthy, S. Gulati, R. Caro Repetto, and X. Serra, "Saraga: Open datasets for research on Indian art music," Empirical Musicology Review, vol. 16, no. 1, 2021.
