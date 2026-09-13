# ROADMAP

```
[x] allow `--harvest sets/` to find all crops recursively
[x] write the margin/fpr ladder to the trained toml
[x] manually labeled go4 are visible in the verdicts tab
[x] verdict items marked 'unclear' are removed from the verdicts tab
[ ] jump to a specific date/time in the crop list
[ ] track near misses eg. 0.01 below the required margin for verdict/alert

[x] add active_hours for the stream; when the stream is inactive, we should
    close the rtsp channels and harvest and recording should also be inactive
[x] add time restrictions for crop harvest and event recording
[x] freeze new crops/events when not at the top of the page
[x] keep the top bar in sight on the webui at all times
[x] expand crops/verdicts and events to fill the screen (while maintaining aspect ratio) when clicked.
[x] for crops/verdicts, allow scroll zoom
[x] add a refresh button to present more items to label in the labeling/training
    web interface without starting the training process; when the training
    button is pressed, train on all that have been labeled
[x] while the learn/train service is running, continuously ingest new crops for labeling
[x] make it possible to toggle show/hide bounding boxes in the webui
[x] the stream preview in the webui should expand to fill the window (preserving aspect ratio)
[x] unload the models while the stream is shut, for a deployment that wants the memory back overnight
[x] say so in the live preview when the stream is closed by [stream] active_hours,
    rather than holding the last frame
[x] windows finer than one per day: weekdays, or more than one window a day
```
