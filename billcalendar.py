#import utilities for date/time and data storage
import datetime
import json
import calendar as cal
import enum
import time

calendar = cal.Calendar(6)

class event:
    def __init__(self, date_and_time, title = "Untitled Event", desc = "No Description", repeat_duration=1, repeat_time_in_weeks=1):
        self.date_and_time = date_and_time
        self.repeat_duration=repeat_duration
        self.repeat_time_in_weeks=repeat_time_in_weeks
        self.title=title
        self.desc=desc
    def isOnDate(self, date):
        # Only match by date (ignore time)
        return date == self.date_and_time

# Store events as a list of event objects
events = []

def add_event(date_str, title, desc):
    # date_str: 'YYYY-MM-DD'
    y, m, d = map(int, date_str.split('-'))
    dt = datetime.date(y, m, d)
    events.append(event(dt, title=title, desc=desc))

def remove_event(date_str, idx):
    # Remove the idx-th event on that date
    y, m, d = map(int, date_str.split('-'))
    dt = datetime.date(y, m, d)
    # Find all events for that date
    day_events = [i for i, e in enumerate(events) if e.date_and_time == dt]
    if 0 <= idx < len(day_events):
        del events[day_events[idx]]

def return_dateData(month, year):
    weeks = calendar.monthdatescalendar(year, month)
    outputData = [[None]*7 for i in range(len(weeks))]
    for i in range(len(weeks)):
        for j in range(7):
            cell_date = weeks[i][j]
            tempStore = []
            for item in events:
                if item.isOnDate(cell_date):
                    tempStore.append((item.title, item.desc))
            # Add date string for JS (YYYY-MM-DD)
            outputData[i][j]=[cell_date.day, tempStore, cell_date.isoformat()]
    return outputData

def get_calendar_data(month, year):
    return return_dateData(month, year)






