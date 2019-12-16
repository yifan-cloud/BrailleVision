/* this file just has code for mode-switching, which will be copied into a larger nrf control file */

typedef enum Mode { depth, objectRecog, textDetect };

Mode mode = depth;
char *mode_names[3] = { "depth", "objectRecog", "textDetect" };
uint8_t up = 0; //whether dial moved up or down; TODO: replace w/ actual dial code

void switch_mode()
{
    //on dial change
    //TODO: get dial change, and which direction
    
    int new_mode;
    if (up)
    {
        //increase mode num
        new_mode = (int) mode + 1 % 3;
    }
    else //down
    {
        //decrease mode num
        new_mode = (int) mode + 2 % 3; // +2 instead of -1 to ensure result always positive
    }
    mode = (Mode) new_mode;

    //actually it'd be much easier to just send 0, 1, or 2
    //cast mode to uint8_t and send it as a single byte
}