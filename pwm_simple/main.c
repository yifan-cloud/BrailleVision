/**
 * Copyright (c) 2015 - 2017, Nordic Semiconductor ASA
 * 
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form, except as embedded into a Nordic
 *    Semiconductor ASA integrated circuit in a product or a software update for
 *    such product, must reproduce the above copyright notice, this list of
 *    conditions and the following disclaimer in the documentation and/or other
 *    materials provided with the distribution.
 * 
 * 3. Neither the name of Nordic Semiconductor ASA nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 * 
 * 4. This software, with or without modification, must only be used with a
 *    Nordic Semiconductor ASA integrated circuit.
 * 
 * 5. Any software provided in binary form under this license must not be reverse
 *    engineered, decompiled, modified and/or disassembled.
 * 
 * THIS SOFTWARE IS PROVIDED BY NORDIC SEMICONDUCTOR ASA "AS IS" AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NORDIC SEMICONDUCTOR ASA OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 */
/** @file
 * @defgroup pwm_example_main main.c
 * @{
 * @ingroup pwm_example
 *
 * @brief PWM Example Application main file.
 *
 * This file contains the source code for a sample application using PWM.
 */

#include <stdio.h>
#include <string.h>
#include "nrf_drv_pwm.h"
#include "app_util_platform.h"
#include "app_error.h"
#include "boards.h"
#include "bsp.h"
#include "nrf_drv_clock.h"
#include "nrf_delay.h"
#include "nrf_gpio.h"
//#include "app_uart.h"
//#include "nrf_delay.h"
//#include "nrf.h"
//#if defined (UART_PRESENT)
//#include "nrf_uart.h"
//#endif
//#if defined (UARTE_PRESENT)
//#include "nrf_uarte.h"
//#endif



#define OUTPUT_PIN_1 17
#define OUTPUT_PIN_2 18
#define OUTPUT_PIN_3 19
#define OUTPUT_PIN_4 20

#define PWM_1_0 NRF_GPIO_PIN_MAP(0,22)
#define PWM_1_1 NRF_GPIO_PIN_MAP(0,23)

#define PWM_2_0 NRF_GPIO_PIN_MAP(0,24)
#define PWM_2_1 NRF_GPIO_PIN_MAP(0,25)

#define PWM_3_0 NRF_GPIO_PIN_MAP(0,9)
#define PWM_3_1 NRF_GPIO_PIN_MAP(0,10)

#define PWM_4_0 NRF_GPIO_PIN_MAP(0,7)
#define PWM_4_1 NRF_GPIO_PIN_MAP(0,8)

#define SELECT_IN NRF_GPIO_PIN_MAP(0,13)
#define RIGHT_IN NRF_GPIO_PIN_MAP(0,14)
#define LEFT_IN NRF_GPIO_PIN_MAP(0,15)

#define SELECT_OUT NRF_GPIO_PIN_MAP(0,3)
#define RIGHT_OUT NRF_GPIO_PIN_MAP(0,4)
#define LEFT_OUT NRF_GPIO_PIN_MAP(0,28)

//#define OUTPUT_PIN_5 15
//#define OUTPUT_PIN_6 16
//#define OUTPUT_PIN_7 17
//#define OUTPUT_PIN_8 18
//#define OUTPUT_PIN_9 19
//#define OUTPUT_PIN_10 20
//#define OUTPUT_PIN_11 21
//#define OUTPUT_PIN_12 22


static nrf_drv_pwm_t m_pwm0 = NRF_DRV_PWM_INSTANCE(0);
//static nrf_drv_pwm_t m_pwm1 = NRF_DRV_PWM_INSTANCE(1);
//static nrf_drv_pwm_t m_pwm2 = NRF_DRV_PWM_INSTANCE(2);


//static nrf_drv_pwm_t m_pwm3 = NRF_DRV_PWM_INSTANCE(3);

// Declare variables holding PWM sequence values. In this example only one channel is used 
nrf_pwm_values_individual_t seq_values1[] = {0, 0, 0, 0};
//nrf_pwm_values_individual_t seq_values2[] = {0, 0, 0, 0};
//nrf_pwm_values_individual_t seq_values3[] = {0, 0, 0, 0};

nrf_pwm_sequence_t const seq1 =
{
    .values.p_individual = seq_values1,
    .length          = NRF_PWM_VALUES_LENGTH(seq_values1),
    .repeats         = 0,
    .end_delay       = 0
};
//
//nrf_pwm_sequence_t const seq2 =
//{
//    .values.p_individual = seq_values2,
//    .length          = NRF_PWM_VALUES_LENGTH(seq_values1),
//    .repeats         = 0,
//    .end_delay       = 0
//};
//
//nrf_pwm_sequence_t const seq3 =
//{
//    .values.p_individual = seq_values3,
//    .length          = NRF_PWM_VALUES_LENGTH(seq_values1),
//    .repeats         = 0,
//    .end_delay       = 0
//};


// Set duty cycle between 0 and 100%
void pwm_update_duty_cycle(uint8_t *duty_cycle)
{
    
    // Check if value is outside of range. If so, set to 100%
    if(duty_cycle[0]>=100)
    {
    seq_values1->channel_0 = 100;
    
    }
    else
    {
    seq_values1->channel_0 = duty_cycle[0];
    }

    if(duty_cycle[1]>=100)
    {
    seq_values1->channel_1 = 100;
    
    }
    else
    {
    seq_values1->channel_1 = duty_cycle[1];
    }

    if(duty_cycle[2]>=100)
    {
    seq_values1->channel_2 = 100;
    
    }
    else
    {
    seq_values1->channel_2 = duty_cycle[2];
    }

    if(duty_cycle[3]>=100)
    {
    seq_values1->channel_3 = 100;
    }
    else
    {
    seq_values1->channel_3 = duty_cycle[3];
    }

//
//        seq_values2->channel_0 = duty_cycle[4];
//        seq_values2->channel_1 = duty_cycle[5];
//        seq_values2->channel_2 = duty_cycle[6];
//        seq_values2->channel_3 = duty_cycle[7];
//
//        seq_values3->channel_0 = duty_cycle[8];
//        seq_values3->channel_1 = duty_cycle[9];
//        seq_values3->channel_2 = duty_cycle[10];
//        seq_values3->channel_3 = duty_cycle[11];


    
    
    nrf_drv_pwm_simple_playback(&m_pwm0, &seq1, 1, NRF_DRV_PWM_FLAG_LOOP);
//    nrf_drv_pwm_simple_playback(&m_pwm1, &seq2, 1, NRF_DRV_PWM_FLAG_LOOP);
//    nrf_drv_pwm_simple_playback(&m_pwm2, &seq3, 1, NRF_DRV_PWM_FLAG_LOOP);
}

static void pwm_init(void)
{

    
    nrf_drv_pwm_config_t const config0 =
    {
        .output_pins =
        {
            OUTPUT_PIN_1, // channel 0
            OUTPUT_PIN_2,             // channel 1
            OUTPUT_PIN_3,             // channel 2
            OUTPUT_PIN_4,
        },
        .irq_priority = APP_IRQ_PRIORITY_LOWEST,
        .base_clock   = NRF_PWM_CLK_1MHz,
        .count_mode   = NRF_PWM_MODE_UP,
        .top_value    = 100,
        .load_mode    = NRF_PWM_LOAD_INDIVIDUAL,
        .step_mode    = NRF_PWM_STEP_AUTO
    };

//     nrf_drv_pwm_config_t const config1 =
//    {
//        .output_pins =
//        {
//            OUTPUT_PIN_5, // channel 0
//            OUTPUT_PIN_6,             // channel 1
//            OUTPUT_PIN_7,             // channel 2
//            OUTPUT_PIN_8,
//        },
//        .irq_priority = APP_IRQ_PRIORITY_LOWEST,
//        .base_clock   = NRF_PWM_CLK_1MHz,
//        .count_mode   = NRF_PWM_MODE_UP,
//        .top_value    = 100,
//        .load_mode    = NRF_PWM_LOAD_INDIVIDUAL,
//        .step_mode    = NRF_PWM_STEP_AUTO
//    };
//
//     nrf_drv_pwm_config_t const config2 =
//    {
//        .output_pins =
//        {
//            OUTPUT_PIN_9, // channel 0
//            OUTPUT_PIN_10,             // channel 1
//            OUTPUT_PIN_11,             // channel 2
//            OUTPUT_PIN_12,
//        },
//        .irq_priority = APP_IRQ_PRIORITY_LOWEST,
//        .base_clock   = NRF_PWM_CLK_1MHz,
//        .count_mode   = NRF_PWM_MODE_UP,
//        .top_value    = 100,
//        .load_mode    = NRF_PWM_LOAD_INDIVIDUAL,
//        .step_mode    = NRF_PWM_STEP_AUTO
//    };
    // Init PWM without error handler
    APP_ERROR_CHECK(nrf_drv_pwm_init(&m_pwm0, &config0, NULL));
//    APP_ERROR_CHECK(nrf_drv_pwm_init(&m_pwm1, &config1, NULL));
//    APP_ERROR_CHECK(nrf_drv_pwm_init(&m_pwm2, &config2, NULL));
    
}

//#define MAX_TEST_DATA_BYTES     (15U)                /**< max number of test bytes to be used for tx and rx. */
//#define UART_TX_BUF_SIZE 256                         /**< UART TX buffer size. */
//#define UART_RX_BUF_SIZE 256                         /**< UART RX buffer size. */
//
//void uart_error_handle(app_uart_evt_t * p_event)
//{
//    if (p_event->evt_type == APP_UART_COMMUNICATION_ERROR)
//    {
//        APP_ERROR_HANDLER(p_event->data.error_communication);
//    }
//    else if (p_event->evt_type == APP_UART_FIFO_ERROR)
//    {
//        APP_ERROR_HANDLER(p_event->data.error_code);
//    }
//}
//
//
//#ifdef ENABLE_LOOPBACK_TEST
///* Use flow control in loopback test. */
//#define UART_HWFC APP_UART_FLOW_CONTROL_ENABLED
//
///** @brief Function for setting the @ref ERROR_PIN high, and then enter an infinite loop.
// */
//static void show_error(void)
//{
//
//    bsp_board_leds_on();
//    while (true)
//    {
//        // Do nothing.
//    }
//}
//
//
///** @brief Function for testing UART loop back.
// *  @details Transmitts one character at a time to check if the data received from the loopback is same as the transmitted data.
// *  @note  @ref TX_PIN_NUMBER must be connected to @ref RX_PIN_NUMBER)
// */
//static void uart_loopback_test()
//{
//    uint8_t * tx_data = (uint8_t *)("\r\nLOOPBACK_TEST\r\n");
//    uint8_t   rx_data;
//
//    // Start sending one byte and see if you get the same
//    for (uint32_t i = 0; i < MAX_TEST_DATA_BYTES; i++)
//    {
//        uint32_t err_code;
//        while (app_uart_put(tx_data[i]) != NRF_SUCCESS);
//
//        nrf_delay_ms(10);
//        err_code = app_uart_get(&rx_data);
//
//        if ((rx_data != tx_data[i]) || (err_code != NRF_SUCCESS))
//        {
//            show_error();
//        }
//    }
//    return;
//}
//#else
///* When UART is used for communication with the host do not use flow control.*/
//#define UART_HWFC APP_UART_FLOW_CONTROL_DISABLED
//#endif

int main(void)
{
    nrf_gpio_cfg_input(PWM_1_0,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_1_1,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_2_0,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_2_1,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_3_0,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_3_1,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_4_0,NRF_GPIO_PIN_PULLDOWN);
    nrf_gpio_cfg_input(PWM_4_1,NRF_GPIO_PIN_PULLDOWN);

    nrf_gpio_cfg_input(SELECT_IN,NRF_GPIO_PIN_PULLUP);
    nrf_gpio_cfg_input(RIGHT_IN,NRF_GPIO_PIN_PULLUP);
    nrf_gpio_cfg_input(LEFT_IN,NRF_GPIO_PIN_PULLUP);

    nrf_gpio_cfg_output(SELECT_OUT);
    nrf_gpio_cfg_output(RIGHT_OUT);
    nrf_gpio_cfg_output(LEFT_OUT);



    // Start clock for accurate frequencies
    NRF_CLOCK->TASKS_HFCLKSTART = 1; 

    // Wait for clock to start
    uint8_t* duty = (uint8_t *)malloc(sizeof(uint8_t) * 4);
    while(NRF_CLOCK->EVENTS_HFCLKSTARTED == 0) 
        ;
    
    pwm_init();


//    while (true)
//    {
//        double haptic_arr[16];
//        char arr[50];
//        uint8_t cr;
//        bool mode = true;
//        if (mode)
//        {
//            //read in haptic array
//            for (int i = 0; i < 16; i++)
//            {
//                //read in one space-separated token
//                int arr_idx = 0;
//                
//                uint8_t done = 0;
//                
//                while (!done)
//                {
//                    while (app_uart_get(&cr) != NRF_SUCCESS);
//                    while (app_uart_put(cr) != NRF_SUCCESS);
// 
//                    if (cr == ' ')
//                    {
//                        //end of token
//                        arr[arr_idx] = '\0';
//                        arr_idx = 0;
//
//                        haptic_arr[i] = strtod(arr, NULL);
//
//                        done = 1;
//                    }
//                    else
//                    {
//                        arr[arr_idx] = cr;
//                        arr_idx++;
//                    }
//                }
//                
//                //restart at beginning of char array
//                //read in a char at a time until a space
//                //upon space, put null char in string
//                //call atof(?) to get float value
//           }
//        }
//        pwm_update_duty_cycle(haptic_arr);
//    }

    for (;;)
    {   
        if(nrf_gpio_pin_read(SELECT_IN))
        {
        nrf_gpio_pin_clear(SELECT_OUT);
        }
        else
        {
        nrf_gpio_pin_set(SELECT_OUT);
        }

        if(nrf_gpio_pin_read(LEFT_IN))
        {
        nrf_gpio_pin_clear(LEFT_OUT);
        }
        else
        {
        nrf_gpio_pin_set(LEFT_OUT);
        }

        if(nrf_gpio_pin_read(RIGHT_IN))
        {
        nrf_gpio_pin_clear(RIGHT_OUT);
        }
        else
        {
        nrf_gpio_pin_set(RIGHT_OUT);
        }
        duty[0] = 33*(2*nrf_gpio_pin_read(PWM_1_0) + nrf_gpio_pin_read(PWM_1_1));
        duty[1] = 33*(2*nrf_gpio_pin_read(PWM_2_0) + nrf_gpio_pin_read(PWM_2_1));
        duty[2] = 33*(2*nrf_gpio_pin_read(PWM_3_0) + nrf_gpio_pin_read(PWM_3_1));
        duty[3] = 33*(2*nrf_gpio_pin_read(PWM_4_0) + nrf_gpio_pin_read(PWM_4_1));
        pwm_update_duty_cycle(duty);
        
    }
}


/** @} */
