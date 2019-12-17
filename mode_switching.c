/* this file just has code for mode-switching, which will be copied into a larger nrf control file
 * based on the pin_change_int example from the nRF5 SDK
 */

#include <stdbool.h>
#include "nrf.h"
#include "nrf_drv_gpiote.h"
#include "app_error.h"
#include "boards.h"

/* pin numbers from pin assignments doc in the Google Drive */
#define ROTARY_RIGHT 14
#define ROTARY_LEFT 15
#define SELECT_BUTTON 13

typedef enum Mode:uint8_t { depth, objectRecog, textDetect };

Mode mode = depth;
uint8_t toSend = 0; //value to send
uint8_t sendMsg = 0; //flag for whether to send a message to Pi

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

void in_pin_handler(nrf_drv_gpiote_pin_t pin, nrf_gpiote_polarity_t action)
{
    switch(pin)
    {
        case ROTARY_LEFT:
            //mode down
            switch_mode(0);
            toSend = (uint8_t) mode;
            sendMsg = 1;
            break;
        case ROTARY_RIGHT:
            //mode up
            switch_mode(1);
            toSend = (uint8_t) mode;
            sendMsg = 1;
            break;
        case SELECT_BUTTON:
            //send 4
            toSend = 4;
            sendMsg = 1;
            break;
    }
}
/**
 * @brief Function for configuring pins
 * configures GPIOTE to give an interrupt on pin change.
 */
static void gpio_init(void)
{
    ret_code_t err_code;

    err_code = nrf_drv_gpiote_init();
    APP_ERROR_CHECK(err_code);

    //TODO: I don't know if we're doing the hacky wiring individual pins directly thing
    // nrf_drv_gpiote_out_config_t out_config = GPIOTE_CONFIG_OUT_SIMPLE(false);

    // err_code = nrf_drv_gpiote_out_init(PIN_OUT, &out_config);
    // APP_ERROR_CHECK(err_code);

    //configure input pins
    nrf_drv_gpiote_in_config_t in_config = GPIOTE_CONFIG_IN_SENSE_LOTOHI(true); //TODO: is lotohi correct?
    in_config.pull = NRF_GPIO_PIN_PULLUP;

    err_code = nrf_drv_gpiote_in_init(ROTARY_LEFT, &in_config, in_pin_handler);
    APP_ERROR_CHECK(err_code);

    nrf_drv_gpiote_in_event_enable(ROTARY_LEFT, true);

    err_code = nrf_drv_gpiote_in_init(ROTARY_RIGHT, &in_config, in_pin_handler);
    APP_ERROR_CHECK(err_code);

    nrf_drv_gpiote_in_event_enable(ROTARY_RIGHT, true);

    err_code = nrf_drv_gpiote_in_init(SELECT_BUTTON, &in_config, in_pin_handler);
    APP_ERROR_CHECK(err_code);

    nrf_drv_gpiote_in_event_enable(SELECT_BUTTON, true);
}

/**
 * @brief Function for application main entry.
 */
int main(void)
{
    gpio_init();

    while (true)
    {
        //if new message to send, send update message to RPi
        if (sendMsg)
        {
            //TODO: ...whatever this turns out to be

            sendMsg = 0;
        }
    }
}
