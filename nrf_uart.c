#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include "app_uart.h"
#include "app_error.h"
#include "nrf_delay.h"
#include "nrf.h"
#include "bsp.h"
#include "nrf_uart.h"
#include "nrf_uarte.h"

#define MAX_TEST_DATA_BYTES     (15U)                /**< max number of test bytes to be used for tx and rx. */
#define UART_TX_BUF_SIZE 256                         /**< UART TX buffer size. */
#define UART_RX_BUF_SIZE 256                         /**< UART RX buffer size. */

uint8_t onDepthMode = 0;

void uart_error_handle(app_uart_evt_t * p_event)
{
    if (p_event->evt_type == APP_UART_COMMUNICATION_ERROR)
    {
        APP_ERROR_HANDLER(p_event->data.error_communication);
    }
    else if (p_event->evt_type == APP_UART_FIFO_ERROR)
    {
        APP_ERROR_HANDLER(p_event->data.error_code);
    }
}

/**
 * @brief Function for main application entry.
 */
int main(void)
{
    uint32_t err_code;

    bsp_board_init(BSP_INIT_LEDS);

    const app_uart_comm_params_t comm_params =
      {
          RX_PIN_NUMBER,
          TX_PIN_NUMBER,
          RTS_PIN_NUMBER,
          CTS_PIN_NUMBER,
          UART_HWFC,
          false,
          NRF_UART_BAUDRATE_115200
      };

    APP_UART_FIFO_INIT(&comm_params,
                         UART_RX_BUF_SIZE,
                         UART_TX_BUF_SIZE,
                         uart_error_handle,
                         APP_IRQ_PRIORITY_LOWEST,
                         err_code);

    APP_ERROR_CHECK(err_code);

    printf("\r\nUART example started.\r\n");

    while (true)
    {
        uint8_t cr;
        while (app_uart_get(&cr) != NRF_SUCCESS);
        while (app_uart_put(cr) != NRF_SUCCESS);

        if (cr == 'q' || cr == 'Q')
        {
            printf(" \r\nExit!\r\n");

            while (true)
            {
                // Do nothing.
            }
        }
    }

    while (true)
    {
        double haptic_arr[16];
        char arr[50];
        uint8_t cr;
        if (onDepthMode)
        {
            //read in haptic array
            for (int i = 0; i < 16; i++)
            {
                //read in one space-separated token
                int arr_idx = 0;
                
                uint8_t done = 0;
                
                while (!done)
                {
                    while (app_uart_get(&cr) != NRF_SUCCESS);

                    if (cr == ' ')
                    {
                        //end of token
                        arr[arr_idx] = '\0';
                        arr_idx = 0;

                        haptic_arr[i] = strtod(arr, NULL);

                        done = 1;
                    }
                    else
                    {
                        arr[arr_idx] = cr;
                        arr_idx++;
                    }
                }
                
                //restart at beginning of char array
                //read in a char at a time until a space
                //upon space, put null char in string
                //call atof(?) to get float value
            }

            //control motors
        }
    }

    /*
    haptic array in
    read in a full string. need to know when string ends
    split on spaces
    can probably read in

    mode out
    mode based on dial
    up/down for intensity
    
    gpio
    button in
    button out

    this probably needs a state machine
    so that modes only do what they need
    actually the nrf doesn't do anything for the vision modes
    so just an if keeping track of whether on depth mode or not
    */
}
