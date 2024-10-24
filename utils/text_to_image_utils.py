import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16)

pipe = pipe.to("cuda")

def text_to_image(artist_style, negative_prompt_value, num_inference_steps_value, guidance_scale_value, avatar_or_illustration,seed,lighting,environment,color_scheme,point_of_view,background, art_style,person_group_detail):
  global prompt_text
  if avatar_or_illustration == "Avatar":
    if person_group_detail != "":
      prompt_text = "Selfie of a" + person_group_detail + "for the profil photo of a socialNetwork"
    else :
      prompt_text = "Selfie of a man with brown curly hair, brown eyes for the profil photo of a socialNetwork"
    if artist_style != "":
      prompt_text = prompt_text + "with influence of the artist style of " + artist_style
    if lighting != "":
        prompt_text = prompt_text + "with lighting " + lighting
    if environment != "":
        prompt_text = prompt_text + "in a " + environment
    if color_scheme != "":
        prompt_text = prompt_text + "with color scheme " + color_scheme
    if point_of_view != "":
        prompt_text = prompt_text + "from a " + point_of_view
    if background != "":
        prompt_text = prompt_text + "with background " + background
    if art_style != "":
        prompt_text = prompt_text + "with art style " + art_style

  if avatar_or_illustration == "Illustration":
    if person_group_detail != "":
          prompt_text = "Seflie of " + person_group_detail + " highly detailed face and eyes, for a publication on social networks, 64K, UHD, HDR, large angle, Fish-eye lens, by a professional photograph, vivid colors, bokeh"
    else :
          prompt_text = "Illustration of a party with a group of friends for a publication on social networks"
    if artist_style != "":
        prompt_text = prompt_text + "with influence of the artist style of " + artist_style
    if lighting != "":
        prompt_text = prompt_text + "with lighting " + lighting
    if environment != "":
        prompt_text = prompt_text + "in a " + environment
    if color_scheme != "":
        prompt_text = prompt_text + "with color scheme " + color_scheme
    if point_of_view != "":
        prompt_text = prompt_text + "from a " + point_of_view
    if background != "":
        prompt_text = prompt_text + "with background " + background
    if art_style != "":
        prompt_text = prompt_text + "with art style " + art_style
  if seed == "":
    if avatar_or_illustration == "Avatar":
      if num_inference_steps_value == "":
        if guidance_scale_value == "":
          prompt = prompt_text
          generator = torch.Generator("cuda")
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value).images[0]
          return image, prompt
        else:
          prompt = prompt_text
          generator = torch.Generator("cuda")
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, guidance_scale=guidance_scale_value).images[0]
          return image, prompt
      if guidance_scale_value == "":
        prompt = prompt_text
        generator = torch.Generator("cuda")
        image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value).images[0]
        return image, prompt
      else:
        prompt = prompt_text
        generator = torch.Generator("cuda")
        image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value, guidance_scale=guidance_scale_value).images[0]
        return image, prompt
    if avatar_or_illustration == "Illustration":
        if num_inference_steps_value == "":
          if guidance_scale_value == "":
            prompt = prompt_text
            generator = torch.Generator("cuda")
            image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value).images[0]
            return image, prompt
          else:
            prompt = prompt_text
            generator = torch.Generator("cuda")
            image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, guidance_scale=guidance_scale_value).images[0]
            return image, prompt
        if guidance_scale_value == "":
          prompt = prompt_text
          generator = torch.Generator("cuda")
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value).images[0]
          return image, prompt
        else:
          prompt = prompt_text
          generator = torch.Generator("cuda")
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value, guidance_scale=guidance_scale_value).images[0]
          return image, prompt
  else:
    if avatar_or_illustration == "Avatar":
      if num_inference_steps_value == "":
        if guidance_scale_value == "":
          prompt = prompt_text
          generator = torch.Generator("cuda").manual_seed(seed)
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value).images[0]
          return image, prompt
        else:
          prompt = prompt_text
          generator = torch.Generator("cuda").manual_seed(seed)
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, guidance_scale=guidance_scale_value).images[0]
          return image, prompt
      if guidance_scale_value == "":
        prompt = prompt_text
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value).images[0]
        return image, prompt
      else:
        prompt = prompt_text
        generator = torch.Generator("cuda").manual_seed(seed)
        image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value, guidance_scale=guidance_scale_value).images[0]
        return image, prompt
    if avatar_or_illustration == "Illustration":
        if num_inference_steps_value == "":
          if guidance_scale_value == "":
            prompt = prompt_text
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value).images[0]
            return image, prompt
          else:
            prompt = prompt_text
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, guidance_scale=guidance_scale_value).images[0]
            return image, prompt
        if guidance_scale_value == "":
          prompt = prompt_text
          generator = torch.Generator("cuda").manual_seed(seed)
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value).images[0]
          return image, prompt
        else:
          prompt = prompt_text
          generator = torch.Generator("cuda").manual_seed(seed)
          image = pipe(prompt, generator=generator, negative_prompt= negative_prompt_value, num_inference_steps=num_inference_steps_value, guidance_scale=guidance_scale_value).images[0]
          return image, prompt
