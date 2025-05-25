import React from 'react'
import insta from '../../../assets/insta.svg'
import fb from '../../../assets/fb.svg'
import x from '../../../assets/x.svg'
import Page_right from '../Page_Right/Page_right'


const Page_left = () => {
  return (
    <div className='flex'>
        <div className='flex flex-col items-start max-w-[720px] mt-[20px]'>
            <div className='flex flex-col items-start max-w-[720px] '>
                <h1 className='text-[65px] font-bold text-white ml-30 mt-20 leading-[75px] '>Know Your <span className='text-teal-400'>Land</span>, Grow Your Best.</h1>
                <div className='flex flex-col items-start mt-12.5 ml-30 max-w-[490px] '>
                    <div className='flex items-start text-white rounded-lg text-[15px]'>
                        <div className='flex items-center justify-center mr-5 w-15 h-10 bg-cyan-700 rounded-full text-lg font-semibold '>1</div>
                        <p>Enter your land <span className='text-teal-400'>details</span> like nitrogen, potassium, phosphorus, temperature, humidity, pH, and rainfall.</p>
                    </div>
                    <div className='flex items-start text-white rounded-lg text-[15px] mt-7.5'>
                        <div className='flex items-center justify-center mr-5 w-11.5 h-10 bg-cyan-700 rounded-full text-lg font-semibold'>2</div>
                        <p>Move to the <span className='text-teal-400'>AI chat</span> and enter your question or select a suggested prompt.</p>
                    </div>
                    <div className='flex items-start text-white rounded-lg text-[15px] mt-7.5'>
                        <div className='flex items-center justify-center mr-5 w-11.75 h-10 bg-cyan-700 rounded-full text-lg font-semibold '>3</div>
                        <p>Grownius AI analyzes the data and provides the best <span className='text-teal-400'>crop</span> recommendation.</p>
                    </div>
                </div>
            </div> 
            <div className='px-30 mt-40 flex gap-[45px] mb-[232px] '>
                <img src={insta} alt="" />
                <img src={fb} alt="" />
                <img src={x} alt="" />
            </div>   
        </div>
        <div className='flex ml-[20px] mt-[2px] '>
            <Page_right/>
        </div>
    </div>
  )
}

export default Page_left
