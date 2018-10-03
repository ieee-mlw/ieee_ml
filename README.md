# ieee_ml
A Message from IEEE ComSoC, ITSoC, CIS and Apollo AI:

Post event updates (10/3/2018, 7.15am)
---------------------------------------
Thank you to everyone who attended the workshop. We hope attendees enjoyed the workshop. We appreciate Dr. Kiran Gunnam’s volunteering to teach the course without taking any fees. The event was made possible through TI, IEEE ComSoC, ITSoC, CIS, Apollo AI and all the [volunteers and helpers](https://github.com/ieee-mlw/ieee_ml/blob/master/volunteers.MD). We appreciate your help. The event photos are posted [here](https://github.com/ieee-mlw/ieee_ml/blob/master/event_pictures.MD).

1. We posted an excellent guided review of Tensorflow Visualization, CLI debugger and Tensor board debugger by Dr. Kiran Gunnam at https://youtu.be/XFLbNRu_kb0

2. Tensorflow_Viualization_Debugging_Manual.pdf is updated to show a simple matrix of which files work with what debuggers and visualizer at the end of the file. Also step-by-step screenshots of Tensorboard debugger are added. Please note that you need to wait for the log files to be created before running the visualizer and some programs will take more time to generate the log files. 

3. Presentation slides are updated on google drive link. Workshop 1 slides now have backup slides on kernel tricks and decision trees. Workshop 2 slide has one minor type 3/0 is corrected to 30 on one of slide.

4. If you still have any issues in running the software, please send us an email on google groups. We will continue to monitor this as we would like the attendees to be able to experiment and understand the software.

5. You can run the software files from this repo in https://colab.research.google.com/ if you do not want to handle the installation issue. Please see readme_google_colaboratory.MD in the home directory of this repo for further instructions. 

6. We sent a simple 2-minute to 5-minute survey asking feedback of 125 attendees on how we can improve future offerings of this course. 
We got a good ~80% of 125 attendees responded to our surveys. (i.e 101 responses).

~80% of responded attendees (80 out of 101 responses) rated the event as very good or excellent.
-----------------------------------------------------------------------
~90% of responded attendees (90 out of 101 responses) rated the event as good or very good or excellent.
---------------------------------------------------------------------
[60% of responded attendees (60 out of 101 responses) rated the event as excellent and 20% of responded attendees (20 out of 101 responses) rated the event as very good. 10% of responded attendees (10 out of 101 responses) rated the event as good.  7% of responded attendees (7 out of 101 responses) rated the event as fair. 
3% of responded attendees (3 out of 101 responses) rated the event as poor.]

On improving the next offering of the course (How could future events be improved? Select all that apply.), 
80% of responded attendees suggested Hosting the event with WiFI so that code can be run at https://colab.research.google.com/ without installation issues.

50% of responded attendees suggested that we need to have the course software pushed to github 2 weeks ahead of the course.

55% of responded attendees suggested to keep the speaker  on time by taking the questions from attendees at the end of each course module. 

27% responded attendees suggested a 5-day boot camp to cover material, software  more in-depth along with a design challenge project.

33% of responded attendees suggested a 2-day workshop than current 2-evening workshop to cover this material along with the software

We will follow this feedback and improve our next offerings.Out of 88 survey respondents, Chris Byrne, Uday Prabhune,Charles McDonald,Matthew Clapp and Peter Nuth won and received one $50 Amazon gift card each. 
------------------------------------------------------------------------------------------------------------------------


Thank you again for making this event successful.

IEEE ML workshop team.

Pre-event Messages
------------------------------------------------

--------------------------------------------------------
Instructions for 2018 IEEE Machine Learning Workshop attendees.
---------------------------------------------------------------
Thank you all for signing up for the IEEE Machine Learing workshops at Texas Instruments( Silicon Valley Auditorium (Building E), 2900 Semiconductor Dr ,Santa Clara, CA 95051).
Please check the event page for schedule and the detailed map of the location: https://www.eventbrite.com/e/ieee-workshops-on-machine-learning-convolutional-neural-networks-and-tensorflow-tickets-45668033317

Please make sure to have the latest repo. More changes related to documentation were done on 9/22 10.56pm.(please scroll down for the changes).
-----------------------------------------------------------------------------------------------------------------------------------
A.Logistics: 
------------------------------------------

1. All the attendees are requested to arrive before 4pm on 9/24 and 9/25 to ensure smooth check-in. The workshop starts at 4pm on each day.We are expecting 125+ attendees on-site and there would be WebEx avaiilable for 
Texas Instruments Employees from remote sites. 
2. Snacks will be provided at the start of the event on each day.(4pm to 4.15pm)
Sandwiches will be provided for dinner on Mon (~6.30pm to 6.50pm).
Pizza will be provided for dinner on Tue. (~6.30pm to 6.50pm).
Vegetarian diet(sandwiches or pizzas based on the day) is also provided in additition to meat based diet.

3. Course slides are sent as pdf files on Friday afternoon through eventbrite. Please contact us if you have not received the slides. 

*****4. If you are attending day 2 of workshop to do interactive programming exercsises, 
you must prepare the laptop with the setup in advance before the workshop based 
on the following instructions in  sections B & C below..****

5. Lot of attendees have this qustion: I purchased a two-day ticket and it mentions Workshops 1 and 2. However, the ticket mentions only 1 day(either 9/24 or 9/25). Is the ticket valid for both days?

IEEE ML team: Sorry for the confusion-it has to do with how eventbrite handles multi-day events. Yes, the 2-day ticket is valid for both the days.
Workshop 1 is on Monday and Workshop 2 is on Tuesday. 
If you purchase Workshop 1 ticket from either day listed on eventbrite, you will need to attend on Monday.
If you purchase Workshop 2 ticket from either day listed on eventbrite, you will need to attend on Tuesday. (this is your case)





B. WiFi and Power Sockets
---------------------------------------
1. There is no availability of WiFi and power sockets for attendees in TI auditorium.
Please bring a fully charged laptop along with the software installed as per section C below.
Espcially on day-2, your laptop should last atleast for 2 hours for the interactive exercises.

(We have limited WiFi connectivity and access to power sockets in the auditorium-this is mainly for 
5 course volunteers who would be helping the attendees. If you need to use WiFi on laptop for any purpose,
you may want to tether to your mobile phone's hotspot.However please **do not count** on installing the software packages 
while on mobile tethering as the software packages for Python and tensorflow are **very large**. These best can be installed
while you are on high speed connection at work or at home.)


C. This Important notice is applicable for people who are attending Day 2 of the workshop.
----------------------------------------------------------------------------------------------------------------------------

2. You must prepare the laptop with the setup in advance before the workshop based on the following instructions.

Simple Setup
-----------
The setup is infact quite simple if you have a latest Tensorflow installation 
and you may not need to do all the instructions.Only additional thing to 
do download the course materials and repo and then do install additional packages using
pip install -r requirements.txt  
Please make sure that you can run the course programs from Jupyter notebook to ensue you have right setup. If in doubt,
please do the detailed setup as instructed below.

Detailed Setup
--------------- 
Follow the instructions based on your preference:
Repo home: https://github.com/ieee-mlw/ieee_ml

Install with Windows: See Windows_README.MD

Install with Linux: See Linux_README.MD

Install with Docker on Windows or Linux: See Docker_README.MD

Install with MacOS: See Mac_README.MD

This process installs several python and tensorflow packages using pip command or docker command. 
For this you need internet connection.
Also you should download the repo as the repo has local data so that the program can be run without 
needing network connection to get the data.
Please make sure that you can run these programs from Jupyter notebook.

 ***********************************************************************************************
 You ***MUST*** complete the install process and downloading of course materials before coming to the Day *1* of workshop . 
 ***********************************************************************************************

Note that laptop with software is not needed on Day 1. It is only needed on Day 2 of workshop.
In case of any issues, you can be helped on Day 1 for Day 2 of workshop.

Help on Setup
-----------------------------------

3. What to do if you face setup issues before the workshop:
The setup is in fact quite simple if you have a latest Tensorflow installation. 
Only additional thing to do download the course materials and repo and then do install additional packages using
pip install -r requirements.txt  

In any case, if you encounter any issues,

Help is available through Google Groups on 9/20 to 9/25: If you face any issues with the setup, please join the Google Groups
https://groups.google.com/forum/#!forum/ieee-machine-learning-workshop
and ask your question there, if it is not already answered. 
We have 8 team members including Apollo AI team members to help on this. 
We also have 4 team members who can help through remote google hangouts.
But please do not wait till the last minute.

You can also reach us directly through eventbrite contact page or 
the course email address(though which we reached out to all the attendees).


4. It is important for all the attendees to come prepared with a working installation so that we focus on the programming and the concepts. For attendees who still have setup issues and whose issues were not resolved through Google Groups, we will do our best to help.

For attendees who registered for both days: Limited installation help is available on the first evening of the workshop 
For attendees who registered for 2nd day: Very limited installation help is available on the second evening of the workshop 
It is very difficult to handle the installation issue if all 125+ attendees simply show up to the workshop without any installation preparation on thier side. So please do your part and try to get the installation done before coming to the workshop. This way, we can really focus on helping the people who may have real and hard problem in getting their installation right.

Documentation and Debugger Improvements 
--------------------------------------------------------------
We also added some more optional files such as screenshot of tensorboard debugger and the expected results so that it is easy for attendees to follow. Tensorflow_Visualisation_Debugging_Manual.pdf, Tensorflow_tutorial.pdf and the cheatsheet-python-grok.pdf as well as simple programs are in the basics directory. We also updated few software files related to Tensorboard debugger as well as expected results as screenshots compiled into PDF files into guided examples and problem sets directorires. If you are using Docker image, please make sure to pull the latest Docker image(kgunnamieee/mlw) which is updated using these repo changes.
We also posted an excellent guided review of Tensorflow Visualization, CLI debugger and Tensor board debugger by Dr. Kiran Gunnam at https://youtu.be/XFLbNRu_kb0


Thank you again for your understanding, we look forward to provide you a great learning experience.

IEEE ML workshop team.

