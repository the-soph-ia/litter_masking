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

# Different Approaches
### Iteration 1:
I used a simple color mask through OpenCV to filter out certain colors from the image. I focused on filtering out white-ish tones since that's the color of most cigarettes, plastics, and paper trash, so even getting that much would be useful. This model worked decently, though it was definitely better suited for high-contrast environments. I did not add any way to track the object's location, so the only use at this stage is in visual recognition, not tracking.

### Iteration 2:
In an attempt to make the litter detection more dynamic relative to the ground around it, I tried creating a graph of the hue values throughout each image frame. By finding the peaks in the graph, I could isolate chunks of color at a time. However, this process ended up making the code run much slower with minimal to no benefit relative to the previous method. While in theory it should be possible to tweak this method and make it work better, especially for detecting different types of trash that aren't white, there are a lot of parameters to consider; it would probably be better to move on to contouring and focus on how to gather the object's position at any given time. 