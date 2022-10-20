import pandas as pd
samples = pd.read_csv('test.csv')

startTime = min(samples['StartTime'])
allEntries = sum(samples['EntriesProcessed'])
samples['StartTime'] -= startTime
samples['EndTime'] -= startTime
samples['Duration'] = samples['EndTime'] - samples['StartTime']
samples['EntriesPercentProc'] = samples['EntriesProcessed']/allEntries*100

import altair as alt
import altair_saver

chart = alt.Chart(samples[['Slot','ThreadID','CoreID',
    'StartTime','EndTime','EntriesProcessed', 'EntriesPercentProc', 'Duration']]).mark_bar(
    cornerRadiusTopLeft=2,
    cornerRadiusTopRight=2,
    cornerRadiusBottomLeft=2,
    cornerRadiusBottomRight=2
    ).encode(
    x='StartTime',
    x2='EndTime',
    y='ThreadID',
    tooltip=['Slot', 'CoreID', 'EntriesProcessed', 'EntriesPercentProc', 'Duration']
).properties(width=1200).interactive()

chart.save('taskstream.html')