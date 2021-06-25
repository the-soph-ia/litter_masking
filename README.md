# Litter Masking
A method to make litter stand out in real-time video feeds.

### Litter has worsened since the start of the pandemic:
https://time.com/5949983/trash-pandemic/
- Portland, OR: 50% increase in residential trash
- Pennsylvania: illegal dumping up 213%
- waterways in America: 26 billion pieces of litter in 2020
- highways in America: 24 billion pieces of litter in 2020

### Why?
- Fewer sanitation workers + higher rates of COVID among them
- More disposable packaging (mail-order boxes, plastic cutlery, takeout boxes, etc.)
- People were worried about other things so focused less on preventing litter

### What does this code do?
- Takes a video as input and outputs a mask of that video that displays litter present
- Uses the very rudimentary model of searching for a range of colors within the image:
- - Cigarettes are the #1 most common type of litter: most are white, which are easy to detect
- Works best where there is more contrast between the ground and the litter
- - This could be improved with a different model of detection, using contouring in OpenCV or perhaps a neural network

### So this is a starting point...
- If we can help computers see trash, we can design robots to pick it up.
- Things to add in the future:
- - Get object's position relative to the camera
- - Improve detection algorithm
- - Add detection for multi-colored or unique pieces of trash
