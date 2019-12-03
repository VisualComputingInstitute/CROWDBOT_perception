#!/usr/bin/env python
import os
import sys
import time

import message_filters
import rospy
from frame_msgs.msg import TrackedPersons

import numpy


def newTracksAvailable(trackedPersons):
    if not stopReceiving:
        global trackIDs, numCycles, occludedTrackCounts, trackCounts

        numCycles += 1
        trackCounts.append(len(trackedPersons.tracks))
        for trackedPerson in trackedPersons.tracks:
            trackIDs.add(trackedPerson.track_id)
        occludedTracks = filter(lambda person: person.is_occluded,
                                trackedPersons.tracks)
        occludedTrackCounts.append(len(occludedTracks))


if __name__ == '__main__':
    maxNumOccludedTracks = 0
    numOccludedTracks = 0
    numCycles = 0
    trackIDs = set()
    trackCounts = []
    occludedTrackCounts = []
    stopReceiving = False

    rospy.init_node("dataset_stats")
    trackedPersonsTopic = rospy.resolve_name("/groundtruth/tracked_persons")

    rospy.loginfo("Listening for tracked persons on " + trackedPersonsTopic +
                  ". Press CTRL+C to stop listening!")

    # Listen for groundtruth tracked persons
    trackedPersonsSubscriber = rospy.Subscriber(trackedPersonsTopic,
                                                TrackedPersons,
                                                newTracksAvailable,
                                                queue_size=1000)
    rospy.spin()
    stopReceiving = True

    trackCountsArray = numpy.array(trackCounts)
    occludedTrackCountsArray = numpy.array(occludedTrackCounts)

    rospy.loginfo(
        "### Recorded %d cycles with %d unique tracks, of which max. %d were visible at the same time! Average: %f Median: %f###"
        % (numCycles, len(trackIDs), numpy.max(trackCountsArray),
           numpy.average(trackCountsArray), numpy.median(trackCountsArray)))
    rospy.loginfo(
        "### Recorded %d cycles with %d occluded track frames, of which max. %d were occluded at the same time! Average: %f Median: %f ###"
        % (numCycles, numpy.sum(occludedTrackCountsArray),
           numpy.max(occludedTrackCountsArray),
           numpy.average(occludedTrackCountsArray),
           numpy.median(occludedTrackCountsArray)))
    trackFile = open('dataset_stats.csv', 'w')
    cycle = 0
    for trackCount in trackCounts:
        trackFile.write("%05d    %.5f   %d\n" %
                        (cycle, cycle / float(len(trackCounts)), trackCount))
        cycle += 1
    trackFile.close()
