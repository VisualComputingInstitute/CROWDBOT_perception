//
//    Copyright (c) 2009-2011
//      Patrick Sudowe	<sudowe@umic.rwth-aachen.de>
//      RWTH Aachen University, Germany
//
//    This file is part of groundHOG.
//
//    GroundHOG is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    GroundHOG is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with groundHOG.  If not, see <http://www.gnu.org/licenses/>.
//

// ------------------------------------------------------------------------------------
// WINDOWS VERSION
// ------------------------------------------------------------------------------------
#ifdef WIN32
#include <time.h>
#include <Windows.h>
#include <mmsystem.h>

typedef struct {
	DWORD start;
	DWORD end;
} Timer;


// ------------------------------------------------------------------------------------
// LINUX VERSION 
// ------------------------------------------------------------------------------------
#else

#include <sys/time.h>

typedef struct {
	timespec start;
	timespec end;
} Timer;

// NOTE: you need to link with -lrt when using this (on Linux / Unix)

#endif

// ------------------------------------------------------------------------------------

extern void startTimer(Timer*);
extern void stopTimer(Timer*);
extern double getTimerValue(Timer*);
