/* this file just has code for mode-switching, which will be copied into a larger nrf control file
 * based on https://github.com/NordicPlayground/nrf52-drv-gpio-example/blob/master/drv_gpio_example.c
 */

#include "pca10040.h"
#include "nrf_error.h"
#include "drv_gpio.h"
#include "nrf_delay.h"

#include <string.h>

#define ROTARY_RIGHT 14
#define ROTARY_LEFT 15
#define SELECT_BUTTON 13

#define M_INPUT_PINS_MSK   ((1UL << ROTARY_RIGHT) | (1UL << ROTARY_LEFT) | (1UL << SELECT_BUTTON))

typedef enum Mode:uint8_t { depth, objectRecog, textDetect };

Mode mode = depth;

void switch_mode(uint8_t changeIsUp)
{
    //on dial change    
    int new_mode;
    if (changeIsUp)
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
}

int main()
{
    drv_gpio_inpin_cfg_t in_cfg   = DRV_GPIO_INPIN_CFG_DEFAULT;

    /* Tweak the default to use the internal pullup resistors of the nRF52 since
       there are no external pullup resistors on the nRF52 development board. */
    in_cfg.pull = DRV_GPIO_PULL_UP;

    drv_gpio_inpin_cfg(ROTARY_LEFT, in_cfg, DRV_GPIO_NO_PARAM_PTR);
    drv_gpio_inpin_cfg(ROTARY_RIGHT, in_cfg, DRV_GPIO_NO_PARAM_PTR);
    drv_gpio_inpin_cfg(SELECT_BUTTON, in_cfg, DRV_GPIO_NO_PARAM_PTR);

    uint8_t sendMsg; //whether to send a message
    uint8_t toSend = 0;
    do
    { 
        uint8_t level;

        sendMsg = 0;
        
        if (drv_gpio_inpin_get(ROTARY_LEFT, &level) == NRF_SUCCESS)
        {
            if (level == DRV_GPIO_LEVEL_LOW) //TODO: should this be low?
            {
                //mode down
                switch_mode(0);
                toSend = (uint8_t) mode;
                sendMsg = 1;
            }
        }
        else if (drv_gpio_inpin_get(ROTARY_RIGHT, &level) == NRF_SUCCESS)
        {
            if (level == DRV_GPIO_LEVEL_LOW)
            {
                //mode up
                switch_mode(1);
                toSend = (uint8_t) mode;
                sendMsg = 1;
            }
        }
        else if (drv_gpio_inpin_get(SELECT_BUTTON, &level) == NRF_SUCCESS)
        {
            if (level == DRV_GPIO_LEVEL_LOW)
            {
                //send 4
                toSend = 4;
                sendMsg = 1;
            }
        }

        //send update message to RPi
        if (sendMsg)
        {
            //TODO: ...whatever this turns out to be
        }
    } while (true) //((drv_gpio_inport_get() & M_INPUT_PINS_MSK) != 0);
    
    //TODO: this is only really relevant if the above loop isn't infinite
    drv_gpio_pin_disconnect(ROTARY_LEFT);
    drv_gpio_pin_disconnect(ROTARY_RIGHT);
    drv_gpio_pin_disconnect(SELECT_BUTTON);
}