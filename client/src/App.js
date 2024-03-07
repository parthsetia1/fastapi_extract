import React,{useState} from 'react';
import axios from 'axios';
import { CircularProgress } from '@mui/material';

const App = () => {
  const [result,setresult]=useState("upload file first");
  const [reload,setreload]=useState(false);
  const [file,setfile]=useState(null);
  const [questions,setquestions]=useState([
    "Do u want to extract complete data of the file?",
    "Do u want to extract metadata of the file?",
    "Do u want to extract locations mentioned in the file?",
    "Do u want to extract headings of the file?",
    "Do u want to extract content under a specific heading of the file?",
    "Do you want summary of pages?"
  ])
  const [ind,setind]=useState(0);
  const [upl,setupl]=useState(false);
  const [heading,setheading]=useState("");
  const [extractionType, setExtractionType] = useState('complete_data');
  const [startp,setstartp]=useState(null);
  const [endp,setendp]=useState(null);
  const [sen,setsen]=useState([]);
  const handlefilesubmit=async ()=>{
    var formData = new FormData();
    formData.append('extract', file);
    setreload(true);
    const ans=await axios.post(`http://127.0.0.1:8000/extract_context`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    setreload(false);
     
     console.log(ans);
     let t="Document_type: "+ans.data.document_type+"  "+"Domain:  "+ans.data.subdomain;
     setresult(t);
     setupl(true);
  }
  const handleyesclick=async()=>{
    var formData = new FormData();
    formData.append('extract', file);
    setreload(true);
    if(ind===0||ind===1)
    {
      
      if(ind===1){
      setExtractionType("metadata");}
      const ans=await axios.post(`http://127.0.0.1:8000/home?extraction_type=${extractionType}`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
     // console.log(ans);
     setreload(false);
     // console.log(ans);
     setresult(ans.data);
     setind(ind+1);
    }
    else if(ind===2)
    {
      const ans=await axios.post(`http://127.0.0.1:8000/extract_locations`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
     // console.log(ans);
     setreload(false);
     // console.log(ans);
     setresult(ans.data);
     setind(ind+1);
    }
    else if(ind===3)
    {
      const ans=await axios.post(`http://127.0.0.1:8000/extract_headings`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
     // console.log(ans);
     setreload(false);
     let a=""
     for(let i=0;i<ans.data.length;i++)
     {
      a+=ans.data[i];
      a+=","
     }
     setresult(a);
     setind(ind+1);
    }
    else if(ind===4)
    {
      formData.append('heading',heading);
      const ans=await axios.post(`http://127.0.0.1:8000/extract_content_headings`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    setreload(false);
    // console.log(ans);
    setresult(ans.data.content);
    setind(ind+1);
    }
    else if(ind===5)
    {
      formData.append("startpage",startp);
      formData.append("endpage",endp);
      const ans=await axios.post(`http://127.0.0.1:8000/extract_summary_pages`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    setreload(false);
    console.log(ans);
    setsen(ans.data.content);
    setresult("below is your summary");
    setind(ind+1);
    }
  }
  return (
    <>
    {upl===false&&(
      <>
      <h1>PDF EXTRACTION</h1>
      <label>Enter file: </label>
    <input type='file' accept='.pdf' onChange={(e)=>{
      setfile(e.target.files[0]);
    }}></input>
    <br></br>
    <br></br>
    <button style={{marginLeft:"5px"}} onClick={handlefilesubmit}>submit</button>
    <br></br>
    <br></br>
    </>
    )}
    {reload===true?(
      <>
      <br></br>
      <CircularProgress />
      </>
    ):(
      <span style={{marginLeft:"5px"}}>{result}</span>
    )}
    {upl===true&&ind<=5&&(
      <>
      <br></br>
      <span>{questions[ind]}</span>
      <br></br>
      {ind===4&&(
        <input type='text' placeholder='heading' onChange={(e)=>{
          setheading(e.target.value);
        }}></input>
      )}
      {ind===5&&(
        <>
        <input type='number' placeholder='startpage' onChange={(e)=>{
          setstartp(e.target.value);
        }}></input>
        <input type='number' placeholder='endpage' onChange={(e)=>{
          setendp(e.target.value);
        }}></input>
        </>
      )}
      {reload===false&&(
        <>
      <button style={{marginLeft:"5px"}} onClick={handleyesclick}>Yes</button>
      <button style={{marginLeft:"5px"}} onClick={()=>{
        setind(ind+1);
        if(ind<=4)
        setresult("Answer the below question:");
      else
      setresult("");
      }}>No</button>
      </>
      )}
      <br></br>
      
      </>
    )}
    {Object.keys(sen).length !== 0 && (
  <>
    {Object.entries(sen).map(([heading, sentences], index) => (
      <div key={index}>
        <h3>{heading}</h3>
        <ul>
          {sentences.map((sentence, i) => (
            <li key={i}>{sentence}</li>
          ))}
        </ul>
      </div>
    ))}
  </>
)}
    {ind>=5&&(
      <>
      <br></br>
      <br></br>
    <span>Thank you</span>
    </>
    )}
    </>
  )
}
export default App;